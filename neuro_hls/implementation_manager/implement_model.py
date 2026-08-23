from collections import defaultdict
import json
import math
from pathlib import Path

from neuro_hls.read_nir import *
from .extract_neuron_param_code import extract_neuron_param_code
from .code_creator import CodeCreator
from .event_graph import build_event_graph

NUM_DASHES_COMMENT = 50
DEFAULT_EVENT_DT = 1e-4
TIME_DRIVEN_DECAY_TOTAL_BITS = 28
TIME_DRIVEN_DYNAMICS_TOTAL_BITS = 52
TIME_DRIVEN_DYNAMICS_INTEGER_BITS = 12
ACTIVE_LIST_SHIFT_TERMS = 4
ACTIVE_LIST_MAX_RIGHT_SHIFT = 31
DEFAULT_ACTIVE_LIST_NOISE_THRESHOLD = 1e-6

TIME_DRIVEN_PARALLELIZABLE_TYPES = (
    Linear, Affine, Conv1d, Conv2d, SumPool2d, AvgPool2d,
    Merge, Flatten, Scale, IF, LIF, CubaLIF,
)

def get_bracket_str(arr):
    if arr is not None:
        return ''.join(f'[{x}]' for x in arr)

    return ""


def _write_time_driven_parallelism_manifest(model, folder_path):
    """Persist requested and effective p values beside generated C++.

    Vitis reports describe the implementation it managed to schedule, whereas
    this file records the static architecture requested from NeuroHLS.  Keeping
    both is essential when a memory dependency changes the achieved II.
    """
    layers = []
    for layer in model.layers:
        if not isinstance(layer, TIME_DRIVEN_PARALLELIZABLE_TYPES):
            continue
        plan = model.resolve_time_driven_parallelism(layer)
        layers.append({
            "name": layer.name,
            "operator": type(layer).__name__,
            "operation_kind": plan.operation_kind,
            "requested_parallelism": plan.requested_parallelism,
            "total_work_items": plan.total_work_items,
            "processing_elements": plan.processing_elements,
            "reuse_cycles": plan.reuse_cycles,
            "effective_parallelism": plan.effective_parallelism,
            "idle_slots": plan.idle_slots,
        })
    payload = {
        "backend": "time-driven",
        "parallelism_contract": "percent_parallel_reuse_v1",
        "layers": layers,
    }
    Path(folder_path, "parallelism_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _time_driven_cuba_lif_params(layer):
    dt = float(layer.dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"time-driven CubaLIF layer {layer.name!r} must use a finite, "
            f"positive dt, got {layer.dt!r}"
        )

    tau_syn = np.asarray(layer.tau_syn, dtype=np.float64)
    tau_mem = np.asarray(layer.tau_mem, dtype=np.float64)
    for name, values in (("tau_syn", tau_syn), ("tau_mem", tau_mem)):
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(
                f"time-driven CubaLIF layer {layer.name!r} has invalid "
                f"{name}; every value must be finite and greater than zero"
            )

    params = layer.get_neuron_params()
    converted = {}
    for name, value in params.items():
        if name == "tau_syn":
            coefficient_name = "alpha_syn"
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                coefficients = dt / tau_syn
            if not np.all(np.isfinite(coefficients)):
                raise ValueError(
                    f"time-driven CubaLIF layer {layer.name!r} produces "
                    f"non-finite {coefficient_name}=dt/tau_syn"
                )
            converted[coefficient_name] = coefficients
        elif name == "tau_mem":
            coefficient_name = "beta_mem"
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                coefficients = dt / tau_mem
            if not np.all(np.isfinite(coefficients)):
                raise ValueError(
                    f"time-driven CubaLIF layer {layer.name!r} produces "
                    f"non-finite {coefficient_name}=dt/tau_mem"
                )
            converted[coefficient_name] = coefficients
        else:
            converted[name] = value
    return converted


def _time_driven_decay_integer_bits(model, coefficient_name):
    maximum = 0.0
    for layer in model.layers:
        if not isinstance(layer, CubaLIF):
            continue
        values = _time_driven_cuba_lif_params(layer)[coefficient_name]
        maximum = max(maximum, float(np.max(values)))

    integer_bits = 1
    if maximum >= 1.0:
        integer_bits = int(math.floor(math.log2(maximum))) + 1
    if integer_bits >= TIME_DRIVEN_DECAY_TOTAL_BITS:
        raise ValueError(
            f"{coefficient_name}={maximum:.17g} does not fit in "
            f"ap_ufixed<{TIME_DRIVEN_DECAY_TOTAL_BITS}, I>"
        )
    return integer_bits


def _event_coordinate_expressions(rank):
    if rank == 1:
        return ("0", "0", "i0")
    if rank == 2:
        return ("0", "i0", "i1")
    if rank == 3:
        return ("i0", "i1", "i2")
    raise ValueError(f"event-driven backend supports only 1D, 2D, or 3D tensors, got rank {rank}")


def _nested_loop_code(shape, body, indent="\t"):
    lines = []
    current_indent = indent
    for index, dimension in enumerate(shape):
        lines.append(f"{current_indent}for (int i{index} = 0; i{index} < {int(dimension)}; ++i{index}) {{")
        current_indent += "\t"
    for line in body:
        lines.append(current_indent + line)
    for _ in shape:
        current_indent = current_indent[:-1]
        lines.append(current_indent + "}")
    return "\n".join(lines) + "\n"


def _resolve_event_dt(model):
    """Return the physical duration, in seconds, of one input step."""
    configured_dt = getattr(model, "event_dt", None)
    if configured_dt is not None:
        event_dt = float(configured_dt)
        if not math.isfinite(event_dt) or event_dt <= 0:
            raise ValueError(f"event_dt must be finite and greater than zero, got {configured_dt!r}")
        return event_dt

    graph_layers = getattr(model, "graph_layers", {}) or {}
    temporal_layers = [
        (name, float(layer.dt))
        for name, layer in graph_layers.items()
        if isinstance(layer, CubaLIF) and hasattr(layer, "dt")
    ]
    if not temporal_layers:
        temporal_layers = [
            (layer.name, float(layer.dt))
            for layer in getattr(model, "layers", ())
            if isinstance(layer, CubaLIF) and hasattr(layer, "dt")
        ]

    for name, layer_dt in temporal_layers:
        if not math.isfinite(layer_dt) or layer_dt <= 0:
            raise ValueError(
                f"event-driven layer {name!r} has an invalid physical dt: {layer_dt!r}"
            )

    if temporal_layers:
        event_dt = temporal_layers[0][1]
        incompatible = [
            (name, layer_dt) for name, layer_dt in temporal_layers[1:]
            if not math.isclose(layer_dt, event_dt, rel_tol=1e-12, abs_tol=0.0)
        ]
        if incompatible:
            details = ", ".join(
                f"{name}={layer_dt:.17g}" for name, layer_dt in temporal_layers
            )
            raise ValueError(
                "event-driven layers must use one shared physical dt; " + details
            )
        return event_dt

    return DEFAULT_EVENT_DT


def _resolve_event_cuba_lif_mode(model):
    configured_mode = getattr(
        model, "event_cuba_lif_mode", "discrete_compatible"
    )
    normalized = str(configured_mode).strip().lower().replace("-", "_")
    aliases = {
        "discrete": "discrete_compatible",
        "step": "discrete_compatible",
        "discrete_compatible": "discrete_compatible",
    }
    if normalized not in aliases:
        raise ValueError(
            "only the 'discrete_compatible' event CubaLIF mode is "
            f"supported; got {configured_mode!r}"
        )
    return aliases[normalized]


def _resolve_event_cuba_lif_strategy(model):
    configured_strategy = getattr(
        model, "event_cuba_lif_strategy", "active_list"
    )
    normalized = str(configured_strategy).strip().lower().replace("-", "_")
    aliases = {
        "active": "active_list",
        "active_list": "active_list",
        "hybrid": "active_list",
        "lightweight_ticks": "active_list",
    }
    if normalized not in aliases:
        raise ValueError(
            "only the 'active_list' event CubaLIF strategy is "
            f"supported; got {configured_strategy!r}"
        )
    return aliases[normalized]


def _resolve_event_active_noise_threshold(model):
    configured_threshold = getattr(
        model,
        "event_active_noise_threshold",
        DEFAULT_ACTIVE_LIST_NOISE_THRESHOLD,
    )
    threshold = float(configured_threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "event active-list noise threshold must be finite and greater "
            f"than zero, got {configured_threshold!r}"
        )
    return threshold


def _active_list_shift_terms(coefficient, coefficient_name, layer_name):
    """Approximate a stable Euler coefficient with four right shifts.

    The first terms use the greedy binary expansion.  The last term is rounded
    to the closest remaining power of two, which materially reduces error for
    coefficients such as 0.1 without introducing a multiplier in hardware.
    """
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient <= 0 or coefficient > 1:
        raise ValueError(
            f"active-list CubaLIF layer {layer_name!r} requires 0 < "
            f"{coefficient_name} <= 1 for shift-add decay, got "
            f"{coefficient!r}"
        )

    residual = coefficient
    shifts = []
    for _ in range(ACTIVE_LIST_SHIFT_TERMS):
        if residual <= 0:
            break
        shift = max(0, int(math.floor(-math.log2(residual))))
        if shift > ACTIVE_LIST_MAX_RIGHT_SHIFT:
            break
        term = math.ldexp(1.0, -shift)
        if term > residual and shifts:
            shift += 1
            term *= 0.5
        elif term > residual:
            shift = int(math.ceil(-math.log2(residual)))
            if shift > ACTIVE_LIST_MAX_RIGHT_SHIFT:
                break
            term = math.ldexp(1.0, -shift)

        is_last = len(shifts) == ACTIVE_LIST_SHIFT_TERMS - 1
        if is_last and residual > 0:
            rounded_shift = int(round(-math.log2(residual)))
            rounded_shift = max(0, min(ACTIVE_LIST_MAX_RIGHT_SHIFT, rounded_shift))
            rounded_term = math.ldexp(1.0, -rounded_shift)
            if sum(math.ldexp(1.0, -item) for item in shifts) + rounded_term <= 1:
                shift = rounded_shift
                term = rounded_term

        shifts.append(shift)
        residual -= term
        if abs(residual) <= 1e-15:
            break

    if not shifts:
        raise ValueError(
            f"active-list CubaLIF layer {layer_name!r} cannot represent "
            f"{coefficient_name}={coefficient:.17g} with right shifts up to "
            f"{ACTIVE_LIST_MAX_RIGHT_SHIFT}"
        )

    approximation = sum(math.ldexp(1.0, -shift) for shift in shifts)
    padded = shifts + [0] * (ACTIVE_LIST_SHIFT_TERMS - len(shifts))
    return np.asarray(padded, dtype=np.uint8), len(shifts), approximation


def _active_list_cuba_lif_params(layer, event_dt):
    """Build multiplier-free tick parameters for one CUBA-LIF layer.

    The stored active current is ``beta * R * u``.  Folding that constant into
    the event-only input gain leaves the per-tick recurrence as two shift-add
    decays and one addition to the membrane potential.
    """
    tau_syn = np.asarray(layer.tau_syn, dtype=np.float64)
    tau_mem = np.asarray(layer.tau_mem, dtype=np.float64)
    r = np.asarray(layer.r, dtype=np.float64)
    w_in = np.asarray(layer.w_in, dtype=np.float64)
    expected_shape = tuple(int(value) for value in layer.input_shape)
    for name, values in (
        ("tau_syn", tau_syn),
        ("tau_mem", tau_mem),
        ("r", r),
        ("w_in", w_in),
    ):
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"active-list CubaLIF layer {layer.name!r} expects {name} "
                f"shape {expected_shape}, got {tuple(values.shape)}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"active-list CubaLIF layer {layer.name!r} has non-finite {name}"
            )
    if np.any(tau_syn <= 0) or np.any(tau_mem <= 0):
        raise ValueError(
            f"active-list CubaLIF layer {layer.name!r} requires positive "
            "tau_syn and tau_mem"
        )

    alpha = event_dt / tau_syn
    beta = event_dt / tau_mem
    neuron_count = int(np.prod(expected_shape))
    alpha_shifts = np.zeros(
        (neuron_count, ACTIVE_LIST_SHIFT_TERMS), dtype=np.uint8
    )
    beta_shifts = np.zeros_like(alpha_shifts)
    alpha_counts = np.zeros(neuron_count, dtype=np.uint8)
    beta_counts = np.zeros(neuron_count, dtype=np.uint8)
    alpha_approx = np.zeros(neuron_count, dtype=np.float64)
    beta_approx = np.zeros(neuron_count, dtype=np.float64)

    for index in range(neuron_count):
        shifts, count, approximation = _active_list_shift_terms(
            alpha.flat[index], "alpha_syn=event_dt/tau_syn", layer.name
        )
        alpha_shifts[index] = shifts
        alpha_counts[index] = count
        alpha_approx[index] = approximation
        shifts, count, approximation = _active_list_shift_terms(
            beta.flat[index], "beta_mem=event_dt/tau_mem", layer.name
        )
        beta_shifts[index] = shifts
        beta_counts[index] = count
        beta_approx[index] = approximation

    # The event path performs the only general multiplication.  At a tick,
    # pending_drive is added after old-current decay, preserving the Euler
    # ordering u_next=(1-alpha)u + alpha*w*input.
    input_gain = (
        alpha_approx.reshape(expected_shape)
        * beta_approx.reshape(expected_shape)
        * r
        * w_in
    )
    return {
        "active_u_shifts": alpha_shifts,
        "active_u_terms": alpha_counts,
        "active_v_shifts": beta_shifts,
        "active_v_terms": beta_counts,
        "v_leak": np.asarray(layer.v_leak),
        "v_threshold": np.asarray(layer.v_threshold),
        "v_reset": np.asarray(layer.v_reset),
        "active_input_gain": input_gain,
    }


def _active_list_sparse_synapse_params(layer):
    """Convert a dense Linear/Affine matrix to input-major CSC storage."""
    weights = np.asarray(layer.weight)
    if weights.ndim != 2:
        raise ValueError(
            f"active-list sparse layer {layer.name!r} requires a 2D weight "
            f"matrix, got shape {tuple(weights.shape)}"
        )

    output_count, input_count = weights.shape
    column_offsets = [0]
    row_indices = []
    nonzero_values = []
    for input_index in range(input_count):
        for output_index in range(output_count):
            value = weights[output_index, input_index]
            if value != 0:
                row_indices.append(output_index)
                nonzero_values.append(value)
        column_offsets.append(len(row_indices))

    # Standard C++ does not permit a zero-sized array.  Empty matrices retain
    # one unreachable storage entry while every CSC column remains empty.
    if not nonzero_values:
        row_indices.append(0)
        nonzero_values.append(weights.dtype.type(0))

    params = {
        "sparse_col_ptr": np.asarray(column_offsets, dtype=np.uint32),
        "sparse_row_idx": np.asarray(row_indices, dtype=np.uint32),
        "sparse_values": np.asarray(nonzero_values, dtype=weights.dtype),
    }
    if isinstance(layer, Affine):
        params["sparse_bias"] = np.asarray(layer.bias)
    return params


def _active_list_sparse_storage_size(layer):
    weights = np.asarray(layer.weight)
    return max(1, int(np.count_nonzero(weights)))


def _event_layer_call(
    layer, input_stream, output_stream, params, reset_argument, instance_id=0,
    step_accumulation=False,
    active_noise_threshold=DEFAULT_ACTIVE_LIST_NOISE_THRESHOLD,
):
    if isinstance(layer, (Linear, Affine)):
        template_values = [
            int(layer.input_shape[0]), int(layer.output_shape[0])
        ]
        template_values.append(_active_list_sparse_storage_size(layer))
        template = "<" + ",".join(str(value) for value in template_values) + ">"
    elif isinstance(layer, Flatten):
        if len(layer.input_shape) != 3:
            raise ValueError("event-driven Flatten supports only a 3D input")
        template = "<" + ",".join(str(int(value)) for value in layer.input_shape) + ">"
    elif isinstance(layer, Conv2d):
        values = []
        for value in layer.get_template_args().values():
            values.extend(value if isinstance(value, tuple) else (value,))
        values.extend(layer.input_shape)
        values.append(layer.output_shape[0])
        template = "<" + ",".join(str(int(value)) for value in values) + ">"
    elif isinstance(layer, SumPool2d):
        values = []
        for value in layer.get_template_args().values():
            values.extend(value if isinstance(value, tuple) else (value,))
        values.extend(layer.input_shape)
        template = "<" + ",".join(str(int(value)) for value in values) + ">"
    elif isinstance(layer, (IF, LIF)):
        if len(layer.input_shape) not in (1, 3):
            raise ValueError(f"event-driven {type(layer).__name__} supports only 1D or 3D tensors")
        values = [int(value) for value in layer.input_shape]
        values.append(instance_id)
        template = "<" + ",".join(str(value) for value in values) + ">"
    elif isinstance(layer, CubaLIF):
        if len(layer.input_shape) != 1:
            raise ValueError("event-driven CubaLIF currently supports only 1D tensors")
        reset_by_subtraction = "true" if layer.reset_by_subtraction else "false"
        values = [1, 1, int(layer.input_shape[0]), instance_id,
                  reset_by_subtraction]
        template = "<" + ",".join(str(value) for value in values) + ">"
    else:
        raise ValueError(
            f"layer {layer.name!r} ({type(layer).__name__}) is not supported by the event-driven backend"
        )

    arguments = [input_stream, output_stream]
    if isinstance(layer, CubaLIF):
        ordered_names = (
            "active_u_shifts", "active_u_terms",
            "active_v_shifts", "active_v_terms",
            "v_leak", "v_threshold", "v_reset", "active_input_gain",
        )
        arguments.extend(params[name] for name in ordered_names)
        arguments.append(format(active_noise_threshold, ".17g"))
        arguments.append(reset_argument)
    elif isinstance(layer, LIF):
        ordered_names = ("tau", "r", "v_leak", "v_threshold", "v_reset")
        arguments.extend(params[name] for name in ordered_names)
        arguments.extend(("event_tau_t(NEURO_HLS_EVENT_DT)", reset_argument))
    elif isinstance(layer, IF):
        arguments.extend(params[name] for name in ("r", "v_threshold", "v_reset"))
        arguments.append(reset_argument)
    elif isinstance(layer, Affine):
        arguments.extend(
            params[name] for name in (
                "sparse_col_ptr", "sparse_row_idx", "sparse_values",
                "sparse_bias",
            )
        )
        arguments.append(reset_argument)
    elif isinstance(layer, Linear):
        arguments.extend(
            params[name] for name in (
                "sparse_col_ptr", "sparse_row_idx", "sparse_values",
            )
        )
    else:
        arguments.extend(params.values())
    if isinstance(layer, CubaLIF):
        function_name = "CubaLIFActiveList"
    elif isinstance(layer, (Linear, Affine)):
        function_name = type(layer).__name__ + "Sparse"
        if step_accumulation:
            function_name += "Step"
    else:
        function_name = type(layer).__name__
        if step_accumulation and isinstance(layer, (Linear, Affine)):
            function_name += "Step"
    return f"\t{function_name}{template}({', '.join(arguments)});\n"


def implement_event_driven_model(model, folder_path, use_float=False):
    event_graph = build_event_graph(model)
    event_dt = _resolve_event_dt(model)
    _resolve_event_cuba_lif_mode(model)
    cuba_lif_strategy = _resolve_event_cuba_lif_strategy(model)
    active_noise_threshold = _resolve_event_active_noise_threshold(model)
    event_dt_literal = format(event_dt, ".17g")
    decay_approximation = getattr(model, "event_decay_approximation", "piecewise_linear")
    if decay_approximation != "piecewise_linear":
        raise ValueError(
            "TS-EFA has been retired; only 'piecewise_linear' is supported, "
            f"got {decay_approximation!r}"
        )
    graph_layers = model.graph_layers
    output_predecessors = event_graph.incoming("output")
    if not output_predecessors:
        raise ValueError("event-driven graph output has no incoming edge")
    for edge in output_predecessors:
        if not graph_layers[edge.source].emits_spike:
            raise ValueError("every layer connected to the event-driven output must emit spikes")

    # The current recurrent adapter materializes exactly one vector at the
    # next watermark.  Reject graphs for which applying that adapter at the
    # producer would either leave an old timestamp on feedback or collapse a
    # simultaneous feed-forward branch.
    for edge in event_graph.feedback_edges:
        producer = graph_layers[edge.source]
        if not isinstance(producer, (Linear, Affine)):
            raise ValueError(
                "event-driven feedback currently requires a Linear or "
                f"Affine producer, got {type(producer).__name__} "
                f"on edge {edge.source!r} -> {edge.target!r}"
            )
        if any(
            not outgoing.feedback
            for outgoing in event_graph.outgoing(edge.source)
        ):
            raise ValueError(
                "event-driven feedback producers cannot also drive a "
                f"feed-forward edge: {edge.source!r}"
            )

    def safe_name(name):
        return "".join(character if character.isalnum() else "_" for character in name)

    edge_streams = {}
    for index, edge in enumerate(event_graph.edges):
        prefix = "feedback" if edge.feedback else "edge"
        edge_streams[edge] = f"{prefix}_{index}_{safe_name(edge.source)}_to_{safe_name(edge.target)}"
    for index, edge in enumerate(event_graph.feedback_edges):
        edge_streams[edge] = f"feedback_state_{index}"
    feedback_write_streams = {
        edge: f"feedback_next_{index}"
        for index, edge in enumerate(event_graph.feedback_edges)
    }

    # The event-driven top-level ports are streams.  Bind the graph boundary
    # directly to those ports so conversion is performed by the testbench.
    input_edges = event_graph.outgoing("input")
    output_edges = event_graph.incoming("output")
    if len(input_edges) == 1:
        edge_streams[input_edges[0]] = "input_stream"
    if len(output_edges) == 1:
        edge_streams[output_edges[0]] = "output_stream"

    producer_streams = {}
    split_remainders = defaultdict(list)
    merge_streams = defaultdict(list)
    for node in event_graph.nodes:
        outgoing = event_graph.outgoing(node)
        if len(outgoing) == 1:
            edge = outgoing[0]
            producer_streams[node] = feedback_write_streams.get(
                edge, edge_streams[edge]
            )
        elif len(outgoing) > 1:
            producer_streams[node] = f"node_{safe_name(node)}_out"
            split_remainders[node] = [
                f"split_{safe_name(node)}_{index}"
                for index in range(max(0, len(outgoing) - 2))
            ]
        incoming = event_graph.incoming(node)
        if len(incoming) > 1:
            merge_streams[node] = [
                f"merge_{safe_name(node)}_{index}" for index in range(len(incoming) - 1)
            ]

    def emit_splits(node):
        outgoing = event_graph.outgoing(node)
        if len(outgoing) <= 1:
            return ""
        destinations = [
            feedback_write_streams.get(edge, edge_streams[edge])
            for edge in outgoing
        ]
        current = producer_streams[node]
        lines = []
        for index in range(len(destinations) - 1):
            if index == len(destinations) - 2:
                second = destinations[index + 1]
            else:
                second = split_remainders[node][index]
            lines.append(f"\tSplit({current}, {destinations[index]}, {second});\n")
            current = second
        return "".join(lines)

    def merged_input(node, code_creator):
        incoming = event_graph.incoming(node)
        if len(incoming) == 1:
            return edge_streams[incoming[0]]
        current = edge_streams[incoming[0]]
        for index, edge in enumerate(incoming[1:]):
            result = merge_streams[node][index]
            code_creator.add_code(f"\tMerge({current}, {edge_streams[edge]}, {result});\n")
            current = result
        return current

    prototype = (
        "void snn_to_hls(hls::stream<ed_spike_t>& input_stream, "
        "hls::stream<ed_spike_t>& output_stream, bool reset_potentials)"
    )
    model_cpp = CodeCreator(folder_path)
    model_h = CodeCreator(folder_path)
    neuron_params_h = CodeCreator(folder_path)
    quantization_h = CodeCreator(folder_path)
    model_h.add_code(f"#define NEURO_HLS_EVENT_DT {event_dt_literal}\n")
    if cuba_lif_strategy == "active_list":
        model_h.add_code(
            "#define NEURO_HLS_EVENT_CUBA_LIF_ACTIVE_LIST 1\n"
            f"#define NEURO_HLS_ACTIVE_NOISE_THRESHOLD "
            f"{format(active_noise_threshold, '.17g')}\n"
        )
    model_h.add_code(f"{prototype};\n")

    def node_event_capacity(node):
        layer = graph_layers[node]
        shape = getattr(layer, "output_shape", None)
        if shape is None:
            shape = getattr(layer, "input_shape", model.output_shape)
        return int(np.prod(shape)) + 1

    max_event_burst = max(
        node_event_capacity(node)
        for node in event_graph.nodes
        if node != "output"
    )

    feedback_parameters = "".join(
        f", hls::stream<ed_spike_t>& {edge_streams[edge]}"
        f", hls::stream<ed_spike_t>& {feedback_write_streams[edge]}"
        for edge in event_graph.feedback_edges
    )
    dataflow_prototype = (
        "static void snn_to_hls_dataflow("
        "hls::stream<ed_spike_t>& input_stream, "
        "hls::stream<ed_spike_t>& output_stream"
        f"{feedback_parameters}, bool reset_potentials)"
    )
    model_cpp.add_code(f"{dataflow_prototype}\n{{\n")
    model_cpp.add_code("\t#pragma HLS DATAFLOW\n")

    for edge, stream_name in edge_streams.items():
        if stream_name in ("input_stream", "output_stream") or edge.feedback:
            continue
        depth = node_event_capacity(edge.source)
        model_cpp.add_code(f"\thls::stream<ed_spike_t> {stream_name};\n")
        model_cpp.add_code(
            f"\t#pragma HLS STREAM variable={stream_name} depth={depth}\n"
        )
    temporary_streams = set(producer_streams.values())
    temporary_streams.update(name for values in split_remainders.values() for name in values)
    temporary_streams.update(name for values in merge_streams.values() for name in values)
    temporary_streams.difference_update(edge_streams.values())
    temporary_streams.difference_update(feedback_write_streams.values())
    for stream_name in sorted(temporary_streams):
        model_cpp.add_code(f"\thls::stream<ed_spike_t> {stream_name};\n")
        model_cpp.add_code(
            f"\t#pragma HLS STREAM variable={stream_name} depth={max_event_burst}\n"
        )

    model_cpp.add_code(emit_splits("input"))

    layer_index = 0
    for node in event_graph.schedule:
        if node in ("input", "output"):
            continue
        layer_index += 1
        layer = graph_layers[node]
        layer_params = {}
        if (
            cuba_lif_strategy == "active_list"
            and isinstance(layer, (Linear, Affine))
        ):
            neuron_params = _active_list_sparse_synapse_params(layer)
        elif isinstance(layer, CubaLIF) and cuba_lif_strategy == "active_list":
            neuron_params = _active_list_cuba_lif_params(layer, event_dt)
        else:
            neuron_params = layer.get_neuron_params()
        for name, value in neuron_params.items():
            if isinstance(layer, CubaLIF) and name in ("u_state", "v_state"):
                continue
            parameter_value = np.asarray(value)
            if (
                isinstance(layer, CubaLIF)
                and name == "w_in"
                and cuba_lif_mode == "discrete_compatible"
            ):
                tau_syn = np.asarray(layer.tau_syn, dtype=np.float64)
                if (
                    not np.all(np.isfinite(tau_syn))
                    or np.any(tau_syn <= 0)
                ):
                    raise ValueError(
                        f"event-driven CubaLIF layer {layer.name!r} has "
                        "invalid tau_syn; every value must be finite and "
                        "greater than zero"
                    )
                # Binned inputs represent one Euler interval, whereas the
                # runtime performs an instantaneous jump.  Precompute the
                # effective jump so the hardware still executes u += w.
                parameter_value = (
                    np.asarray(value, dtype=np.float64)
                    * (event_dt / tau_syn)
                )
            param_name = f"{name}_{layer_index}"
            if name in ("sparse_col_ptr", "sparse_row_idx"):
                param_type = "event_index_t"
                value_format = ".0f"
            elif name in ("active_u_shifts", "active_v_shifts"):
                param_type = "event_shift_t"
                value_format = ".0f"
            elif name in ("active_u_terms", "active_v_terms"):
                param_type = "event_shift_count_t"
                value_format = ".0f"
            elif isinstance(layer, (CubaLIF, LIF)) and name in (
                "tau", "tau_syn", "tau_mem"
            ):
                param_type = "event_tau_t"
                value_format = ".10f"
            else:
                param_type = "weight_t"
                value_format = ".10f"
            neuron_params_h.add_code(
                f"{param_type} {param_name}"
                f"{get_bracket_str(parameter_value.shape)} = "
                f"{extract_neuron_param_code(parameter_value, value_format=value_format)};\n\n"
            )
            layer_params[name] = param_name
        input_stream = merged_input(node, model_cpp)
        output_stream = producer_streams[node]
        model_cpp.add_code(
            _event_layer_call(
                layer, input_stream, output_stream, layer_params,
                "reset_potentials", layer_index,
                any(edge.feedback for edge in event_graph.outgoing(node)),
                active_noise_threshold,
            )
        )
        model_cpp.add_code(emit_splits(node))

    final_event_stream = merged_input("output", model_cpp)

    if final_event_stream != "output_stream":
        model_cpp.add_code("\twhile (true) {\n")
        model_cpp.add_code(f"\t\ted_spike_t event = {final_event_stream}.read();\n")
        model_cpp.add_code("\t\toutput_stream.write(event);\n")
        model_cpp.add_code("\t\tif (event.type != ED_TYPE_SPIKE) break;\n\t}\n")
    model_cpp.add_code("}\n")

    model_cpp.add_code(f"\n{prototype}\n{{\n")
    feedback_arguments = []
    feedback_metadata = []
    for feedback_index, edge in enumerate(event_graph.feedback_edges):
        max_events = node_event_capacity(edge.source)
        previous_name = edge_streams[edge]
        next_name = feedback_write_streams[edge]
        state_name = f"feedback_events_{feedback_index}"
        size_name = f"feedback_size_{feedback_index}"
        model_cpp.add_code(
            f"\tstatic ed_spike_t {state_name}[{max_events}];\n"
            f"\tstatic unsigned int {size_name} = 0;\n"
            f"\thls::stream<ed_spike_t> {previous_name};\n"
            f"\t#pragma HLS STREAM variable={previous_name} depth={max_events}\n"
            f"\thls::stream<ed_spike_t> {next_name};\n"
            f"\t#pragma HLS STREAM variable={next_name} depth={max_events}\n"
            f"\tif (reset_potentials) {size_name} = 0;\n"
            f"\tif ({size_name} == 0) {{\n"
            f"\t\ted_spike_t seed = {{}};\n"
            f"\t\tseed.type = ED_TYPE_END_STEP;\n"
            f"\t\t{previous_name}.write(seed);\n"
            f"\t}} else {{\n"
            f"\t\tfor (unsigned int i = 0; i < {size_name}; i++) "
            f"{previous_name}.write({state_name}[i]);\n"
            f"\t}}\n"
        )
        feedback_arguments.extend((previous_name, next_name))
        feedback_metadata.append(
            (max_events, next_name, state_name, size_name, feedback_index)
        )

    call_arguments = ", ".join(
        ["input_stream", "output_stream", *feedback_arguments, "reset_potentials"]
    )
    model_cpp.add_code(f"\tsnn_to_hls_dataflow({call_arguments});\n")

    for max_events, next_name, state_name, size_name, feedback_index in feedback_metadata:
        done_name = f"feedback_done_{feedback_index}"
        end_sample_name = f"feedback_end_sample_{feedback_index}"
        next_size_name = f"feedback_next_size_{feedback_index}"
        event_name = f"feedback_event_{feedback_index}"
        model_cpp.add_code(
            f"\tbool {done_name} = false;\n"
            f"\tbool {end_sample_name} = false;\n"
            f"\tunsigned int {next_size_name} = 0;\n"
            f"\twhile (!{done_name}) {{\n"
            f"\t\ted_spike_t {event_name} = {next_name}.read();\n"
            f"\t\tif ({event_name}.type != ED_TYPE_SPIKE) {{\n"
            f"\t\t\t{event_name}.timestamp += "
            f"(ed_time_step_t)NEURO_HLS_EVENT_DT;\n"
            f"\t\t\t{event_name}.time_step = "
            f"(unsigned int){event_name}.time_step + 1;\n"
            f"\t\t}}\n"
            f"\t\tif ({next_size_name} < {max_events}) "
            f"{state_name}[{next_size_name}++] = {event_name};\n"
            f"\t\t{end_sample_name} = "
            f"{event_name}.type == ED_TYPE_END_SAMPLE;\n"
            f"\t\t{done_name} = {event_name}.type == ED_TYPE_END_STEP || "
            f"{event_name}.type == ED_TYPE_END_SAMPLE;\n"
            f"\t}}\n"
            f"\t{size_name} = {end_sample_name} ? 0 : {next_size_name};\n"
        )
    model_cpp.add_code("}\n")

    # Cuba-LIF time constants are commonly smaller than 1/256.  With the
    # default ap_fixed<16,8> weight type they quantize to zero, and the
    # generated C++ then evaluates dt / tau during CSim.  Keep the same
    # precision policy used by the time-driven backend.
    has_event_driven_cuba_lif = any(
        isinstance(layer, CubaLIF) for layer in model.layers
    )
    if has_event_driven_cuba_lif:
        if model.weight_quantization == (16, 8):
            model.weight_quantization = (24, 8)
        if model.potential_quantization == (16, 8):
            model.potential_quantization = (24, 8)

    input_quantization = "float" if use_float else f"ap_fixed<{model.input_quantization[0]}, {model.input_quantization[1]}>"
    weight_quantization = "float" if use_float else f"ap_fixed<{model.weight_quantization[0]}, {model.weight_quantization[1]}, AP_RND>"
    potential_quantization = "float" if use_float else f"ap_fixed<{model.potential_quantization[0]}, {model.potential_quantization[1]}>"
    temporal_quantization = "float" if use_float else "ap_fixed<32, 8, AP_RND>"
    quantization_h.add_include("ap_int.h")
    quantization_h.add_include("neuro_hls_functions/bit_type.h")
    quantization_h.add_code(f"typedef {input_quantization} input_t;\n")
    quantization_h.add_code(f"typedef {weight_quantization} weight_t;\n")
    quantization_h.add_code(f"typedef {potential_quantization} potential_t;\n")
    quantization_h.add_code(f"typedef {temporal_quantization} temporal_t;\n")
    quantization_h.add_code(f"typedef {temporal_quantization} event_tau_t;\n")
    if cuba_lif_strategy == "active_list":
        quantization_h.add_code("typedef ap_uint<6> event_shift_t;\n")
        quantization_h.add_code("typedef ap_uint<3> event_shift_count_t;\n")
        quantization_h.add_code("typedef ap_uint<32> event_index_t;\n")
    model_h.add_include("neuro_hls_functions/bit_type.h")
    model_h.add_include("quantization.h")
    model_h.add_include("neuro_hls_functions/event_driven.h")
    model_cpp.add_include("neuro_hls_functions/event_driven.h")
    model_cpp.add_include("quantization.h")
    model_cpp.add_include("neuron_params.h")
    model_cpp.add_include("snn_implementation.h")
    neuron_params_h.add_include("quantization.h")
    model_h.create_file("snn_implementation.h")
    model_cpp.create_file("snn_implementation.cpp")
    neuron_params_h.create_file("neuron_params.h")
    quantization_h.create_file("quantization.h")

def _resolve_backend(use_event_driven=False, backend=None):
    if backend is None:
        return "event-driven" if use_event_driven else "time-driven"

    normalized = str(backend).strip().lower().replace("_", "-")
    aliases = {
        "time-driven": "time-driven",
        "time": "time-driven",
        "event-driven": "event-driven",
        "event": "event-driven",
    }
    if normalized not in aliases:
        raise ValueError(
            "backend must be 'time-driven' or 'event-driven', "
            f"got {backend!r}"
        )
    selected = aliases[normalized]
    if use_event_driven and selected != "event-driven":
        raise ValueError(
            "use_event_driven=True conflicts with backend='time-driven'"
        )
    return selected


def _flatten_template_values(template_args):
    values = []
    for value in template_args.values():
        if isinstance(value, (tuple, np.ndarray)):
            values.extend(value)
        else:
            values.append(value)
    return [int(value) for value in values]


def _time_driven_template_values(model, layer):
    """Return structural and percent-parallel reuse arguments in ABI order."""
    if not isinstance(
        layer,
        (
            Linear, Affine, Conv1d, Conv2d, SumPool2d, AvgPool2d,
            Flatten, Scale, IF, LIF, CubaLIF,
        ),
    ):
        # I, LI, and CubaLI deliberately retain their current implementation.
        return _flatten_template_values(layer.get_template_args())

    plan = model.resolve_time_driven_parallelism(layer)

    if isinstance(layer, (Linear, Affine)):
        return [plan.processing_elements]

    if isinstance(layer, (Conv1d, Conv2d, SumPool2d, AvgPool2d)):
        values = _flatten_template_values(layer.get_template_args())
        values.append(plan.processing_elements)
        return values

    if isinstance(layer, (Flatten, Scale, IF, LIF)):
        values = [int(value) for value in layer.input_shape]
        values.append(plan.processing_elements)
        return values

    # CubaLIF has a type-valued first argument and is emitted in its special
    # call path below.
    return []


def _template_suffix(values):
    if not values:
        return ""
    return "<" + ",".join(str(int(value)) for value in values) + ">"


def implement_model(
    model, folder_path, use_float=False, use_event_driven=False, backend=None
):
    selected_backend = _resolve_backend(use_event_driven, backend)

    if selected_backend == "event-driven":
        return implement_event_driven_model(model, folder_path, use_float)

    # (id da camada do NIR, nome a ser usado na impl)
    layer_names = {}

    model_prototype = f"void snn_to_hls(input_t (&input){get_bracket_str(model.input_shape)}, bit_t (&output){get_bracket_str(model.output_shape)}, bool reset_potentials)"

    model_cpp = CodeCreator(folder_path)
    model_h = CodeCreator(folder_path)
    neuron_params_h = CodeCreator(folder_path)
    quantization_h = CodeCreator(folder_path)

    model_h.add_code(f"{model_prototype};")
    model_cpp.add_code(f"{model_prototype}\n{'{'}")

    for (idx, layer) in enumerate(model.layers[1:-1]):

        model_cpp.add_code("\n//" + "-" * NUM_DASHES_COMMENT)
        model_cpp.add_code(f"\n// implementation of '{layer.name}' layer")
        model_cpp.add_code("\n//" + "-" * NUM_DASHES_COMMENT + "\n\n")

        if isinstance(layer, layer_configuration.Merge):

            for (dep_name, is_recurrent) in layer.dependencies:

                if is_recurrent and dep_name not in layer_names:
                    
                    impl_dep_name = f"layer_{len(layer_names) + 1}_rec"
                    layer_names[dep_name] = impl_dep_name

                    rec_layer_output_type = "bit_t" if layer.emits_spike else "potential_t"
                    model_cpp.add_code(f"\tstatic {rec_layer_output_type} {impl_dep_name}{get_bracket_str(layer.input_shape)} = {{}};\n")
                    recurrent_indices = "".join(
                        f"[i{index}]" for index in range(len(layer.input_shape))
                    )
                    model_cpp.add_code("\tif (reset_potentials) {\n")
                    model_cpp.add_code(
                        _nested_loop_code(
                            layer.input_shape,
                            [
                                f"{impl_dep_name}{recurrent_indices} = "
                                f"{rec_layer_output_type}(0);"
                            ],
                            indent="\t\t",
                        )
                    )
                    model_cpp.add_code("\t}\n")

            parallelism = model.resolve_time_driven_parallelism(layer)
            model_cpp.add_code(
                f"\tMerge<{parallelism.processing_elements}>("
                f"{layer_names.get(layer.layer_1, layer.layer_1)}, "
                f"{layer_names.get(layer.layer_2, layer.layer_2)});\n"
            )
            continue

        if layer.name in layer_names:
            cur_layer_id = "_".join(layer_names[layer.name].split("_")[1:])
        else:
            cur_layer_id = len(layer_names) + 1

        if isinstance(layer, CubaLIF):
            neuron_params = _time_driven_cuba_lif_params(layer)
        else:
            neuron_params = layer.get_neuron_params()
        for name, value in neuron_params.items():
            param_type = {
                "alpha_syn": "alpha_syn_t",
                "beta_mem": "beta_mem_t",
                "tau": "temporal_t",
                "tau_syn": "temporal_t",
                "tau_mem": "temporal_t",
            }.get(name, "weight_t")
            value_format = ".17g" if name in ("alpha_syn", "beta_mem") else ".10f"
            neuron_params_h.add_code(
                f"{param_type} {name}_{cur_layer_id}"
                f"{get_bracket_str(value.shape)} = "
                f"{extract_neuron_param_code(value, value_format=value_format)};\n\n"
            )

        input_accum_name = layer_names.get(layer.dependencies[0][0], layer.dependencies[0][0])

        # Declarando potencial da camada, caso ela nao seja recorrente

        if not layer.is_recurrent:

            if idx == len(model.layers) - 3:
                name = "output"
            else:
                name = f"layer_{len(layer_names) + 1}"
                layer_names[layer.name] = name
                output_type = "bit_t" if layer.emits_spike else "potential_t"
                model_cpp.add_code(f"\t{output_type} {name}{get_bracket_str(layer.output_shape)};\n")
        else:
            name = layer_names[layer.name]

        # Chamando a funcao
        neuron_params_call = ", ".join(f"{key}_{cur_layer_id}" for key in neuron_params.keys())
        
        if len(neuron_params_call) > 0: 
            neuron_params_call = ", " + neuron_params_call

        template_args = _template_suffix(
            _time_driven_template_values(model, layer)
        )

        func_name = type(layer).__name__
        # Reduction operators use dedicated flat reuse kernels.  Their last
        # template argument is the exact processing-element count from the
        # time-driven plan, not independent output/reduction lane factors.
        if isinstance(
            layer, (Linear, Affine, Conv1d, Conv2d, SumPool2d, AvgPool2d)
        ):
            func_name += "Reuse"

        if layer.emits_spike: # IF, LIF, CuBa-LIF, etc...

            if isinstance(layer, CubaLIF):
                reset_by_subtraction = "true" if layer.reset_by_subtraction else "false"
                parallelism = model.resolve_time_driven_parallelism(layer)
                model_cpp.add_code(
                    f"\t{func_name}<dynamics_t,{parallelism.processing_elements}>("
                    f"{input_accum_name}, {name}"
                    f"{neuron_params_call}, reset_potentials, "
                    f"{reset_by_subtraction});\n"
                )
            else:
                mem_potentials = f"mem_potentials_{cur_layer_id}"

                model_cpp.add_code(f"\tstatic potential_t {mem_potentials}{get_bracket_str(layer.output_shape)} = {{}};\n")
                # Only the LIF templates take the reset-mode flag; IF shares this
                # branch and its template stops at reset_potentials.
                reset_argument = ""
                if isinstance(layer, LIF):
                    reset_argument = f", {'true' if layer.reset_by_subtraction else 'false'}"
                model_cpp.add_code(f"\t{func_name}{template_args}({input_accum_name}, {name}, {mem_potentials}{neuron_params_call}, reset_potentials{reset_argument});\n")

        else:
            model_cpp.add_code(f"\t{func_name}{template_args}({input_accum_name}, {name}{neuron_params_call});\n")
    
    model_cpp.add_code("}\n")

    quantization_h.add_include("ap_int.h")
    quantization_h.add_include("neuro_hls_functions/bit_type.h")

    has_time_driven_cuba_lif = any(
        isinstance(layer, CubaLIF) for layer in model.layers
    )
    if has_time_driven_cuba_lif:
        if model.weight_quantization == (16, 8):
            model.weight_quantization = (24, 8)
        if model.potential_quantization == (16, 8):
            model.potential_quantization = (24, 8)

    input_quantization = "float" if use_float else f"ap_fixed<{model.input_quantization[0]}, {model.input_quantization[1]}>"
    weight_quantization = "float" if use_float else f"ap_fixed<{model.weight_quantization[0]}, {model.weight_quantization[1]}, AP_RND>"
    potential_quantization = "float" if use_float else f"ap_fixed<{model.potential_quantization[0]}, {model.potential_quantization[1]}>"
    quantization_h.add_code(f"typedef {input_quantization} input_t;\n")
    quantization_h.add_code(f"typedef {weight_quantization} weight_t;\n")
    quantization_h.add_code(f"typedef {potential_quantization} potential_t;\n")
    temporal_quantization = "float" if use_float else "ap_fixed<32, 8, AP_RND>"
    quantization_h.add_code(f"typedef {temporal_quantization} temporal_t;\n")
    if has_time_driven_cuba_lif:
        if use_float:
            alpha_syn_quantization = "float"
            beta_mem_quantization = "float"
            dynamics_quantization = "float"
        else:
            alpha_syn_integer_bits = _time_driven_decay_integer_bits(
                model, "alpha_syn"
            )
            beta_mem_integer_bits = _time_driven_decay_integer_bits(
                model, "beta_mem"
            )
            alpha_syn_quantization = (
                f"ap_ufixed<{TIME_DRIVEN_DECAY_TOTAL_BITS}, "
                f"{alpha_syn_integer_bits}, AP_RND, AP_SAT>"
            )
            beta_mem_quantization = (
                f"ap_ufixed<{TIME_DRIVEN_DECAY_TOTAL_BITS}, "
                f"{beta_mem_integer_bits}, AP_RND, AP_SAT>"
            )
            dynamics_quantization = (
                f"ap_fixed<{TIME_DRIVEN_DYNAMICS_TOTAL_BITS}, "
                f"{TIME_DRIVEN_DYNAMICS_INTEGER_BITS}, AP_RND, AP_SAT>"
            )
        quantization_h.add_code(
            f"typedef {alpha_syn_quantization} alpha_syn_t;\n"
        )
        quantization_h.add_code(
            f"typedef {beta_mem_quantization} beta_mem_t;\n"
        )
        quantization_h.add_code(
            f"typedef {dynamics_quantization} dynamics_t;\n"
        )

    model_h.add_include("neuro_hls_functions/bit_type.h")
    model_h.add_include("quantization.h")

    model_cpp.add_include("neuro_hls_functions/bit_type.h")
    model_cpp.add_include("neuro_hls_functions/time_driven.h")
    model_cpp.add_include("quantization.h")
    model_cpp.add_include("neuron_params.h")
    model_cpp.add_include("snn_implementation.h")

    neuron_params_h.add_include("quantization.h")
    
    model_h.create_file("snn_implementation.h")
    model_cpp.create_file("snn_implementation.cpp")
    neuron_params_h.create_file("neuron_params.h")
    quantization_h.create_file("quantization.h")
    _write_time_driven_parallelism_manifest(model, folder_path)
    
    
