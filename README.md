# NeuroHLS

NeuroHLS translates a supported static NIR spiking-neural-network graph into
synthesizable C++ for FPGA high-level synthesis (HLS).  A single source graph
is lowered through two independent backends, so that dense parallelism and
event sparsity can be compared under the same model, workload, and numerical
configuration:

- **time-driven** evaluates statically shaped tensors once per time step, and
  exposes compile-time parallelism through the parameter `p`;
- **event-driven** uses scalar event streams, FIFOs, watermarks, and HLS
  `DATAFLOW` between actors.

The generator emits an inspectable project: the C++ model, the parameter and
quantization headers, a testbench, and the Tcl scripts.  Parameters, types,
interfaces, and directives stay visible in the generated sources rather than
behind a closed flow.

Generated projects are snapshots: regenerate them after changing the generator
or the runtime headers.  The supported graph subset, the numerical contracts,
and the backend restrictions are reported by the generator itself, which
rejects an unsupported operator or topology instead of silently assigning
different semantics.

## Repository layout

| Path | Contents |
| --- | --- |
| `neuro_hls/` | The compiler: frontend, lowering, code generation, and the tool drivers. |
| `sim/` | Reproducible measurement pipeline over already generated projects. |
| `validation/` | Notebooks and artefacts of the NIR interoperability validation. |
| `tests/` | Unit tests for the frontend, the generators, and both parallelism paths. |
| `MLP_test/`, `SCNN_test/`, `SRNN_test/` | Experiment drivers and derived reports. |
| `nir_examples/` | Sample NIR graphs and datasets. |

The dissertation that documents this artefact lives on the `dissertation`
branch, under `dissertation_mesclado/`.

## Requirements

Python 3.9 or newer, with `nir`, `nirtorch`, `numpy`, and `torch` for graph
handling and reference execution.  Vitis HLS and Vivado are required only to
synthesize, simulate, and measure a generated project; generating one needs
neither.

## Generating a project

```python
from neuro_hls import NeuroHls
from neuro_hls.backend_utils import copy_backend_to

copy_backend_to("my_project")
hls = NeuroHls("my_project")

model = hls.read_nir_file("graph.nir")
model.define_time_driven_parallelism(0.05)
hls.implement_model(model, use_float=False, backend="time-driven")

hls.define_test_dataset(
    "dataset.npz", data_is_binary=True, step_count=32,
    different_sample_per_step=True, max_samples=100,
)
hls.create_testbench(total_samples=100, batch_size=10, reset_potentials=True)
```

Pass `backend="event-driven"` to lower the same graph through the other path.
`NeuroHls` also drives the tools directly, through `run_csim`, `run_synth`,
`run_cosim`, and the export, SAIF, and power-report helpers.

## Measuring a generated project

`sim/` runs the tool flow in isolation.  It copies the input into `sim/runs/`,
so Vitis and Vivado never modify the original project, and then chains CSim,
HLS synthesis, co-simulation, IP export, out-of-context synthesis,
post-synthesis simulation, SAIF capture, and the power report.

```bash
python -m sim run --project my_project
python -m sim status --run <run-id>
./sim/watch_runs.sh          # follow runs executing in parallel
```

Power reports default to vectorless activity; SAIF-annotated capture from the
post-synthesis simulation is available but costly on larger projects.  See
[sim/README.md](sim/README.md) for the stages and the environment file.

## Functional validation

`validation/` reproduces the three interoperability experiments of the NIR
study with NeuroHLS as an additional participant: a single LIF neuron, an SCNN
for N-MNIST, and two CUBA-LIF SRNNs for Braille sequences.

```bash
python validation/run_notebooks.py
```

Each notebook records graph, stimulus, dataset, and result provenance through
hashes, and writes its derived figures and tables to `validation/results/`.

## Time-driven parallelism and reuse

The public time-driven parameter `p` is a normalised percentage of the
operator's static work domain, not a number of output lanes.  For a layer with
`W` statically scheduled operations, NeuroHLS derives:

```text
p = 0       -> U = 1 processing element (explicit serial sentinel)
0 < p <= 1  -> U = min(W, max(1, floor(p * W + 0.5)))
R           = ceil(W / U) reuse groups
p_eff       = U / W
I           = U * R - W idle slots in the final group
```

`Linear`/`Affine`, convolutions, and pooling count MAC or accumulation terms;
element-wise primitives count output elements; neuron primitives count neuron
updates.  The resulting `W`, `U`, `R`, `p_eff`, and `I` are written to
`parallelism_manifest.json` in every generated time-driven project.

```python
model.define_time_driven_parallelism(0.025)
model.define_time_driven_layer_parallelism("linear_1", 0.01)
```

`define_layer_parallelism(...)` remains as a deprecated alias.  There is no
separate reduction-parallelism switch: the single `p` covers the static work
domain of reduction operators too.  HLS may still report a larger initiation
interval than requested because of memory ports, accumulation dependencies,
clock constraints, or routing.

## Event-driven execution

The event-driven backend deliberately has no configurable `p`.  It retains
concurrent actors through `DATAFLOW`, bounded FIFOs, sparse operator logic,
and scalar stream transfers, but does not replicate internal actor lanes from
a global parallelism setting.  Legacy calls with `p=0` issue a deprecation
warning; non-zero event-driven settings fail explicitly.

## Tests

```bash
python -m unittest discover -s tests
python -m unittest tests.test_time_driven_parallelism tests.test_event_driven_parallelism
```

Vitis HLS synthesis and co-simulation remain necessary to establish achieved
II, resource use, timing, and power for a particular target device.
