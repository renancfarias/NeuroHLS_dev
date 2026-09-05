import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuro_hls.implementation_manager.event_graph import build_event_graph
from neuro_hls.implementation_manager.implement_model import implement_model
from neuro_hls.read_nir.get_model_config_from_nir import get_model_config_from_nir
from neuro_hls.read_nir.layer_configuration import (
    CubaLIF, IF, Input, Linear, ModelConfig, Output, Scale,
)


ROOT = Path(__file__).resolve().parents[1]


def recurrent_model():
    model = ModelConfig()
    input_layer = Input("input", np.array([1]))
    feedforward = Linear("feed", (1,), (1,), np.array([[1.0]]))
    neuron = IF(
        "neuron", (1,), (1,), np.array([1.0]),
        np.array([0.5]), np.array([0.0])
    )
    recurrent = Linear("recurrent", (1,), (1,), np.array([[1.0]]))
    output = Output("output", np.array([1]))

    model.graph_layers = {
        layer.name: layer
        for layer in (input_layer, feedforward, neuron, recurrent, output)
    }
    model.graph_edges = [
        ("input", "feed"),
        ("feed", "neuron"),
        ("neuron", "recurrent"),
        ("recurrent", "neuron"),
        ("neuron", "output"),
    ]
    model.input_shape = np.array([1])
    model.output_shape = np.array([1])
    return model


def time_driven_recurrent_model():
    model = ModelConfig()
    model.add_layer(Input("input", np.array([1])))

    feedforward = Linear("feed", (1,), (1,), np.array([[1.0]]))
    feedforward.add_dependency("input", is_recurrent=False)
    model.add_layer(feedforward)

    neuron = IF(
        "neuron", (1,), (1,), np.array([1.0]),
        np.array([0.5]), np.array([0.0])
    )
    neuron.add_dependency("feed", is_recurrent=False)
    neuron.add_dependency("recurrent", is_recurrent=True)
    model.add_layer(neuron)

    recurrent = Linear("recurrent", (1,), (1,), np.array([[1.0]]))
    recurrent.add_dependency("neuron", is_recurrent=False)
    recurrent.is_recurrent = True
    model.add_layer(recurrent)

    readout = Scale("readout", (1,), (1,), np.array([1.0]))
    readout.add_dependency("neuron", is_recurrent=False)
    model.add_layer(readout)

    output = Output("output", np.array([1]))
    output.add_dependency("readout", is_recurrent=False)
    model.add_layer(output)
    return model


def time_driven_cuba_lif_model(tau_syn=0.0002, tau_mem=0.001):
    model = ModelConfig()
    model.add_layer(Input("input", np.array([1])))

    neuron = CubaLIF(
        "neuron", (1,), (1,),
        tau_syn=np.array([tau_syn]),
        tau_mem=np.array([tau_mem]),
        r=np.array([10.0]),
        v_leak=np.array([0.0]),
        v_threshold=np.array([0.4]),
        v_reset=np.array([0.0]),
        w_in=np.array([1.0]),
    )
    neuron.add_dependency("input", is_recurrent=False)
    model.add_layer(neuron)

    output = Output("output", np.array([1]))
    output.add_dependency("neuron", is_recurrent=False)
    model.add_layer(output)
    return model


def active_list_cuba_lif_model(tau_syn=0.0002, tau_mem=0.0004):
    model = ModelConfig()
    input_layer = Input("input", np.array([1]))
    model.add_layer(input_layer)

    linear = Linear("linear", (1,), (1,), np.array([[1.0]]))
    linear.add_dependency("input", is_recurrent=False)
    model.add_layer(linear)

    neuron = CubaLIF(
        "neuron", (1,), (1,),
        tau_syn=np.array([tau_syn]),
        tau_mem=np.array([tau_mem]),
        r=np.array([2.0]),
        v_leak=np.array([0.0]),
        v_threshold=np.array([0.3]),
        v_reset=np.array([0.0]),
        w_in=np.array([1.0]),
    )
    neuron.add_dependency("linear", is_recurrent=False)
    model.add_layer(neuron)

    output = Output("output", np.array([1]))
    output.add_dependency("neuron", is_recurrent=False)
    model.add_layer(output)
    model.graph_layers = {
        layer.name: layer for layer in (input_layer, linear, neuron, output)
    }
    model.graph_edges = [
        ("input", "linear"), ("linear", "neuron"),
        ("neuron", "output"),
    ]
    model.define_event_cuba_lif_strategy("active_list")
    return model


class RecurrentEventGeneratorTests(unittest.TestCase):

    def test_intermediate_graph_identifies_feedback_edge(self):
        graph = build_event_graph(recurrent_model())
        self.assertEqual(graph.recurrent_components, (("neuron", "recurrent"),))
        self.assertEqual(
            [(edge.source, edge.target) for edge in graph.feedback_edges],
            [("recurrent", "neuron")],
        )
        self.assertLess(graph.schedule.index("neuron"), graph.schedule.index("recurrent"))

    def test_intermediate_graph_handles_multiple_recurrent_subgraphs(self):
        model = ModelConfig()
        model.graph_layers = {name: object() for name in (
            "input", "a", "b", "c", "d", "output"
        )}
        model.graph_edges = [
            ("input", "a"), ("a", "b"), ("b", "a"),
            ("a", "c"), ("c", "d"), ("d", "c"),
            ("d", "output"),
        ]
        graph = build_event_graph(model)
        self.assertEqual(
            {(edge.source, edge.target) for edge in graph.feedback_edges},
            {("b", "a"), ("d", "c")},
        )
        self.assertEqual(len(graph.recurrent_components), 2)

    def test_generated_recurrence_uses_split_merge_and_one_step_feedback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(
                recurrent_model(), build_dir, use_float=True,
                use_event_driven=True
            )
            implementation = (build_dir / "snn_implementation.cpp").read_text()
            self.assertIn("static ed_spike_t feedback_events_0[2]", implementation)
            self.assertIn("Split(", implementation)
            self.assertIn("Merge(", implementation)
            self.assertIn(
                "#pragma HLS STREAM variable=feedback_state_0 depth=2",
                implementation,
            )
            self.assertIn("snn_to_hls_dataflow(", implementation)

            driver = build_dir / "driver.cpp"
            driver.write_text(
                '#include <cassert>\n#include "snn_implementation.h"\n'
                'static bool step(hls::stream<ed_spike_t>& in, '
                'hls::stream<ed_spike_t>& out, bool active, bool reset, '
                'int marker_type) { '
                'ed_spike_t e = {}; if (active) { e.type = ED_TYPE_SPIKE; '
                'e.amplitude = 1; in.write(e); } e = {}; '
                'e.type = marker_type; in.write(e); '
                'snn_to_hls(in, out, reset); bool fired = false; while (true) { '
                'e = out.read(); if (e.type != ED_TYPE_SPIKE) break; '
                'fired = true; } return fired; } '
                'int main() { hls::stream<ed_spike_t> in, out; '
                'assert(step(in, out, true, true, ED_TYPE_END_STEP)); '
                'assert(step(in, out, false, false, ED_TYPE_END_SAMPLE)); '
                'assert(!step(in, out, false, false, ED_TYPE_END_STEP)); '
                'return 0; }\n'
            )
            executable = build_dir / "recurrent_model"
            compile_result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    "-I", str(build_dir),
                    str(build_dir / "snn_implementation.cpp"), str(driver),
                    "-o", str(executable),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_generator_rejects_feedback_without_a_safe_step_adapter(self):
        direct = ModelConfig()
        direct_layers = (
            Input("input", np.array([1])),
            Linear("feed", (1,), (1,), np.array([[1.0]])),
            IF(
                "neuron", (1,), (1,), np.array([1.0]),
                np.array([0.5]), np.array([0.0]),
            ),
            Output("output", np.array([1])),
        )
        direct.graph_layers = {layer.name: layer for layer in direct_layers}
        direct.graph_edges = [
            ("input", "feed"), ("feed", "neuron"),
            ("neuron", "feed"), ("neuron", "output"),
        ]
        direct.input_shape = np.array([1])
        direct.output_shape = np.array([1])
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError, "requires a Linear or Affine producer"
            ):
                implement_model(
                    direct, temporary_directory, use_float=True,
                    use_event_driven=True,
                )

        mixed = ModelConfig()
        mixed_layers = (
            Input("input", np.array([1])),
            Linear("feed", (1,), (1,), np.array([[1.0]])),
            IF(
                "neuron", (1,), (1,), np.array([1.0]),
                np.array([0.5]), np.array([0.0]),
            ),
            Linear("recurrent", (1,), (1,), np.array([[1.0]])),
            IF(
                "readout", (1,), (1,), np.array([1.0]),
                np.array([0.5]), np.array([0.0]),
            ),
            Output("output", np.array([1])),
        )
        mixed.graph_layers = {layer.name: layer for layer in mixed_layers}
        mixed.graph_edges = [
            ("input", "feed"), ("feed", "neuron"),
            ("neuron", "recurrent"), ("recurrent", "feed"),
            ("recurrent", "readout"), ("readout", "output"),
        ]
        mixed.input_shape = np.array([1])
        mixed.output_shape = np.array([1])
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError, "cannot also drive a feed-forward edge"
            ):
                implement_model(
                    mixed, temporary_directory, use_float=True,
                    use_event_driven=True,
                )

    def test_time_driven_recurrent_buffer_is_cleared_on_reset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(
                time_driven_recurrent_model(), build_dir, use_float=True,
                backend="time-driven",
            )
            implementation = (build_dir / "snn_implementation.cpp").read_text()
            self.assertIn("static potential_t layer_2_rec[1] = {};", implementation)
            self.assertIn("if (reset_potentials) {", implementation)
            self.assertIn("layer_2_rec[i0] = potential_t(0);", implementation)

            driver = build_dir / "time_driven_recurrent_driver.cpp"
            driver.write_text(
                '#include <cassert>\n#include "snn_implementation.h"\n'
                'int main() { float active[1] = {1}; float silent[1] = {0}; '
                'bit_t output[1] = {}; '
                'snn_to_hls(active, output, true); '
                'assert((unsigned int)output[0] == 1); '
                'snn_to_hls(silent, output, false); '
                'assert((unsigned int)output[0] == 1); '
                'snn_to_hls(silent, output, true); '
                'assert((unsigned int)output[0] == 0); return 0; }\n'
            )
            executable = build_dir / "time_driven_recurrent_model"
            compile_result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    "-I", str(build_dir),
                    str(build_dir / "snn_implementation.cpp"), str(driver),
                    "-o", str(executable),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_time_driven_cuba_lif_generates_alpha_syn_and_beta_mem(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            model = time_driven_cuba_lif_model()
            implement_model(
                model, build_dir, use_float=False, backend="time-driven"
            )

            implementation = (build_dir / "snn_implementation.cpp").read_text()
            parameters = (build_dir / "neuron_params.h").read_text()
            quantization = (build_dir / "quantization.h").read_text()

            self.assertIn("alpha_syn_t alpha_syn_1[1] = {0.5};", parameters)
            self.assertIn("beta_mem_t beta_mem_1[1]", parameters)
            self.assertNotIn("tau_syn_1", parameters)
            self.assertNotIn("tau_mem_1", parameters)
            self.assertIn(
                "CubaLIF<dynamics_t,1>(input, output, alpha_syn_1, "
                "beta_mem_1,",
                implementation,
            )
            self.assertNotIn("weight_t(0.0001)", implementation)
            self.assertIn(
                "typedef ap_ufixed<28, 1, AP_RND, AP_SAT> alpha_syn_t;",
                quantization,
            )
            self.assertIn(
                "typedef ap_ufixed<28, 1, AP_RND, AP_SAT> beta_mem_t;",
                quantization,
            )
            self.assertIn(
                "typedef ap_fixed<52, 12, AP_RND, AP_SAT> dynamics_t;",
                quantization,
            )

            implement_model(
                model, build_dir, use_float=True, backend="time-driven"
            )
            driver = build_dir / "time_driven_decay_driver.cpp"
            driver.write_text(
                '#include <cassert>\n#include "snn_implementation.h"\n'
                'int main() { float input[1] = {1}; bit_t output[1] = {}; '
                'snn_to_hls(input, output, true); '
                'assert((unsigned int)output[0] == 1); return 0; }\n'
            )
            executable = build_dir / "time_driven_decay_model"
            compile_result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    "-I", str(build_dir),
                    str(build_dir / "snn_implementation.cpp"), str(driver),
                    "-o", str(executable),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_time_driven_cuba_lif_rejects_invalid_time_constants(self):
        for tau_syn, tau_mem in ((0.0, 0.001), (0.0002, -1.0)):
            with self.subTest(tau_syn=tau_syn, tau_mem=tau_mem):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(
                        ValueError, "every value must be finite"
                    ):
                        implement_model(
                            time_driven_cuba_lif_model(tau_syn, tau_mem),
                            temporary_directory,
                            use_float=False,
                            backend="time-driven",
                        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "non-finite alpha_syn"):
                implement_model(
                    time_driven_cuba_lif_model(
                        np.nextafter(0.0, 1.0), 0.001
                    ),
                    temporary_directory,
                    use_float=False,
                    backend="time-driven",
                )

    def test_braille_srnn_nir_generates_the_expected_recurrent_subgraph(self):
        model = get_model_config_from_nir(
            str(ROOT / "nir_examples" / "braille_noDelay_bias_zero.nir")
        )
        graph = build_event_graph(model)
        self.assertEqual(
            [(edge.source, edge.target) for edge in graph.feedback_edges],
            [("lif1.w_rec", "lif1.lif")],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(model, build_dir, use_float=True, use_event_driven=True)
            implementation = (build_dir / "snn_implementation.cpp").read_text()
            self.assertEqual(implementation.count("Split("), 1)
            self.assertEqual(implementation.count("Merge("), 1)
            self.assertEqual(
                implementation.count(
                    "#pragma HLS STREAM variable=feedback_state_0 depth=39"
                ), 1
            )

            driver = build_dir / "driver.cpp"
            driver.write_text(
                '#include "snn_implementation.h"\n'
                'static void step(hls::stream<ed_spike_t>& in, '
                'hls::stream<ed_spike_t>& out, bool reset) { '
                'ed_spike_t e = {}; e.type = ED_TYPE_END_STEP; in.write(e); '
                'snn_to_hls(in, out, reset); do { e = out.read(); } '
                'while (e.type == ED_TYPE_SPIKE); } '
                'int main() { hls::stream<ed_spike_t> in, out; '
                'step(in, out, true); step(in, out, false); return 0; }\n'
            )
            executable = build_dir / "braille_srnn"
            compile_result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    "-I", str(build_dir),
                    str(build_dir / "snn_implementation.cpp"), str(driver),
                    "-o", str(executable),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_cuba_lif_keeps_sub_1_over_256_time_constants(self):
        """Fixed-point event generation must not quantize tau values to zero."""
        model = get_model_config_from_nir(
            str(ROOT / "nir_examples" / "braille_noDelay_bias_zero.nir")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(model, build_dir, use_float=False, use_event_driven=True)
            quantization = (build_dir / "quantization.h").read_text()
            parameters = (build_dir / "neuron_params.h").read_text()
            self.assertIn("typedef ap_fixed<24, 8, AP_RND> weight_t;", quantization)
            self.assertIn("typedef ap_fixed<32, 8, AP_RND> event_tau_t;", quantization)
            self.assertIn("active_u_shifts_", parameters)
            self.assertIn("active_v_shifts_", parameters)
            self.assertNotIn("tau_syn_", parameters)
            self.assertNotIn("tau_mem_", parameters)

    def test_generator_uses_physical_timestamps_and_integer_step_indices(self):
        model = get_model_config_from_nir(
            str(ROOT / "nir_examples" / "braille_noDelay_bias_zero.nir")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(model, build_dir, use_float=True, use_event_driven=True)
            implementation = (build_dir / "snn_implementation.cpp").read_text()
            header = (build_dir / "snn_implementation.h").read_text()

            self.assertIn("#define NEURO_HLS_EVENT_DT 0.0001", header)
            self.assertIn("hls::stream<ed_spike_t>& input_stream", header)
            self.assertNotIn("event_time_step", implementation)
            self.assertNotIn("event_timestamp", implementation)

    def test_generator_selects_piecewise_linear_decay_for_cuba_lif(self):
        model = get_model_config_from_nir(
            str(ROOT / "nir_examples" / "braille_noDelay_noBias_subtract.nir")
        )
        model.define_event_decay_approximation("pwl")

        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(model, build_dir, use_float=False, use_event_driven=True)
            implementation = (build_dir / "snn_implementation.cpp").read_text()
            cuba_calls = [
                line for line in implementation.splitlines()
                if "CubaLIFActiveList<" in line
            ]
            self.assertEqual(len(cuba_calls), 2)
            self.assertTrue(all(",true>(" in line for line in cuba_calls))

        with self.assertRaisesRegex(ValueError, "only the 'piecewise_linear'"):
            model.define_event_decay_approximation("ts-efa")
        with self.assertRaisesRegex(ValueError, "only the 'piecewise_linear'"):
            model.define_event_decay_approximation("unknown")

    def test_piecewise_linear_preserves_reset_by_zero_metadata(self):
        model = get_model_config_from_nir(
            str(ROOT / "nir_examples" / "braille_noDelay_bias_zero.nir")
        )
        model.define_event_decay_approximation("piecewise_linear")

        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(model, build_dir, use_float=False, use_event_driven=True)
            implementation = (build_dir / "snn_implementation.cpp").read_text()
            cuba_calls = [
                line for line in implementation.splitlines()
                if "CubaLIFActiveList<" in line
            ]
            self.assertEqual(len(cuba_calls), 2)
            self.assertTrue(all(",false>(" in line for line in cuba_calls))

    def test_decay_selection_is_noop_for_if_only_scnn(self):
        nir_path = str(ROOT / "nir_examples" / "cnn_sinabs.nir")
        default_model = get_model_config_from_nir(nir_path)
        pwl_model = get_model_config_from_nir(nir_path)
        pwl_model.define_event_decay_approximation("piecewise_linear")

        with tempfile.TemporaryDirectory() as default_directory, \
                tempfile.TemporaryDirectory() as pwl_directory:
            implement_model(default_model, default_directory, use_float=False, use_event_driven=True)
            implement_model(pwl_model, pwl_directory, use_float=False, use_event_driven=True)
            for file_name in (
                "snn_implementation.cpp", "neuron_params.h", "quantization.h"
            ):
                self.assertEqual(
                    (Path(default_directory) / file_name).read_bytes(),
                    (Path(pwl_directory) / file_name).read_bytes(),
                )
            implementation = (
                Path(pwl_directory) / "snn_implementation.cpp"
            ).read_text()
            self.assertNotIn("CubaLIF<", implementation)
            self.assertIn("IF<", implementation)

    def test_generator_rejects_incompatible_physical_time_steps(self):
        model = get_model_config_from_nir(
            str(ROOT / "nir_examples" / "braille_noDelay_bias_zero.nir")
        )
        cuba_layers = [
            layer for layer in model.graph_layers.values()
            if isinstance(layer, CubaLIF)
        ]
        self.assertGreaterEqual(len(cuba_layers), 2)
        cuba_layers[1].dt = 2e-4

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "one shared physical dt"):
                implement_model(
                    model, temporary_directory, use_float=True,
                    use_event_driven=True,
                )

    def test_cuba_lif_modes_select_input_jump_and_refractory_contract(self):
        model = time_driven_cuba_lif_model(tau_syn=0.0002, tau_mem=0.001)
        model.graph_layers = {layer.name: layer for layer in model.layers}
        model.graph_edges = [("input", "neuron"), ("neuron", "output")]

        with tempfile.TemporaryDirectory() as discrete_directory:
            implement_model(
                model, discrete_directory, use_float=True,
                use_event_driven=True,
            )
            parameters = (Path(discrete_directory) / "neuron_params.h").read_text()
            implementation = (
                Path(discrete_directory) / "snn_implementation.cpp"
            ).read_text()
            self.assertIn("active_input_gain_1[1]", parameters)
            self.assertIn("CubaLIFActiveList<", implementation)

        with self.assertRaisesRegex(ValueError, "only.*discrete_compatible"):
            model.define_event_cuba_lif_mode("physical")
        with self.assertRaisesRegex(ValueError, "only.*discrete_compatible"):
            model.define_event_cuba_lif_mode("hybrid")

    def test_physical_time_allows_cuba_lif_to_integrate_between_steps(self):
        model = ModelConfig()
        model.add_layer(Input("input", np.array([1])))
        neuron = CubaLIF(
            "neuron", (1,), (1,),
            tau_syn=np.array([0.0002]),
            tau_mem=np.array([0.001]),
            r=np.array([10.0]),
            v_leak=np.array([0.0]),
            v_threshold=np.array([0.7]),
            v_reset=np.array([0.0]),
            w_in=np.array([1.0]),
        )
        neuron.add_dependency("input", is_recurrent=False)
        model.add_layer(neuron)
        output_layer = Output("output", np.array([1]))
        output_layer.add_dependency("neuron", is_recurrent=False)
        model.add_layer(output_layer)
        with self.assertRaisesRegex(ValueError, "only.*discrete_compatible"):
            model.define_event_cuba_lif_mode("continuous")

    def test_active_list_strategy_api_and_validation(self):
        model = ModelConfig()
        self.assertEqual(model.event_cuba_lif_strategy, "active_list")
        model.define_event_cuba_lif_strategy("hybrid")
        self.assertEqual(model.event_cuba_lif_strategy, "active_list")
        model.define_event_active_noise_threshold(2e-5)
        self.assertEqual(model.event_active_noise_threshold, 2e-5)

        with self.assertRaisesRegex(ValueError, "CubaLIF strategy"):
            model.define_event_cuba_lif_strategy("calendar_and_ticks")
        for threshold in (0, -1, float("inf"), float("nan")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "noise threshold"):
                    model.define_event_active_noise_threshold(threshold)

        incompatible = active_list_cuba_lif_model()
        with self.assertRaisesRegex(ValueError, "only.*discrete_compatible"):
            incompatible.define_event_cuba_lif_mode("continuous_physical")

        unstable = active_list_cuba_lif_model(tau_syn=0.00005)
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "0 < alpha_syn"):
                implement_model(
                    unstable, temporary_directory, use_float=True,
                    use_event_driven=True,
                )

    def test_active_list_generator_emits_sparse_shift_add_datapath(self):
        model = active_list_cuba_lif_model()
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(
                model, build_dir, use_float=False, use_event_driven=True
            )
            implementation = (build_dir / "snn_implementation.cpp").read_text()
            parameters = (build_dir / "neuron_params.h").read_text()
            quantization = (build_dir / "quantization.h").read_text()
            header = (build_dir / "snn_implementation.h").read_text()

            self.assertIn("LinearSparse<1,1,1>", implementation)
            self.assertIn("CubaLIFActiveList<1,1,1", implementation)
            self.assertNotIn("CubaLIF<1,1,1", implementation)
            self.assertIn("event_index_t sparse_col_ptr_1[2] = {0, 1};", parameters)
            self.assertIn("event_index_t sparse_row_idx_1[1] = {0};", parameters)
            self.assertIn("event_shift_t active_u_shifts_2[1][4]", parameters)
            self.assertIn("{1, 0, 0, 0}", parameters)
            self.assertIn("event_shift_t active_v_shifts_2[1][4]", parameters)
            self.assertIn("{2, 0, 0, 0}", parameters)
            self.assertIn("active_input_gain_2[1] = {0.2500000000};", parameters)
            self.assertIn("typedef ap_uint<32> event_index_t;", quantization)
            self.assertIn("NEURO_HLS_EVENT_CUBA_LIF_ACTIVE_LIST", header)

    def test_active_list_generated_model_spikes_on_later_lightweight_tick(self):
        model = active_list_cuba_lif_model()
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            implement_model(
                model, build_dir, use_float=True, use_event_driven=True
            )
            driver = build_dir / "active_list_driver.cpp"
            driver.write_text(
                '#include <cassert>\n#include <cmath>\n'
                '#include "snn_implementation.h"\n'
                'static int tick(hls::stream<ed_spike_t>& in, '
                'hls::stream<ed_spike_t>& out, int step, bool input, '
                'bool reset, double* spike_time) { '
                'ed_spike_t e = {}; if (input) { e.type = ED_TYPE_SPIKE; '
                'e.amplitude = 1; e.timestamp = step * NEURO_HLS_EVENT_DT; '
                'e.time_step = step; in.write(e); } e = {}; '
                'e.type = ED_TYPE_END_STEP; '
                'e.timestamp = (step + 1) * NEURO_HLS_EVENT_DT; '
                'e.time_step = step; in.write(e); snn_to_hls(in, out, reset); '
                'int spikes = 0; while (true) { e = out.read(); '
                'if (e.type != ED_TYPE_SPIKE) break; ++spikes; '
                '*spike_time = (double)e.timestamp; } return spikes; } '
                'int main() { hls::stream<ed_spike_t> in, out; double t = -1; '
                'assert(tick(in, out, 0, true, true, &t) == 0); '
                'assert(tick(in, out, 1, false, false, &t) == 1); '
                'assert(std::fabs(t - 0.0002) < 1e-9); '
                'assert(tick(in, out, 0, true, true, &t) == 0); '
                'assert(tick(in, out, 1, false, false, &t) == 1); '
                'return 0; }\n'
            )
            executable = build_dir / "active_list_model"
            compile_result = subprocess.run(
                [
                    "g++", "-std=c++11", "-Wno-unknown-pragmas",
                    "-I", str(ROOT / "tests" / "stubs"),
                    "-I", str(ROOT / "neuro_hls" / "backend"),
                    "-I", str(build_dir),
                    str(build_dir / "snn_implementation.cpp"), str(driver),
                    "-o", str(executable),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)


if __name__ == "__main__":
    unittest.main()
