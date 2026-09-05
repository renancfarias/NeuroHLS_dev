# Reproduction of the NIR interoperability experiments with NeuroHLS

This directory reproduces the LIF, SCNN, and SRNN experiments from Pedersen
et al., *Neuromorphic intermediate representation: A unified instruction set
for interoperable brain-inspired computing*, Nature Communications 15, 8122
(2024), DOI [`10.1038/s41467-024-52259-9`](https://doi.org/10.1038/s41467-024-52259-9).

The official reference code is `neuromorphs/NIR`, tag `v1.0.4`, commit
`8f9177e9f74374de7bd72e49b42217a24df716ec`. The NIR graphs and datasets
already under `nir_examples/` are checked by SHA-256 in the generated
provenance files.

## Notebooks

- `01_lif.ipynb`: the official 1,000-step stimulus, membrane voltage, output
  spikes, timing error, voltage RMSE, and cosine similarity.
- `02_scnn.ipynb`: N-MNIST SCNN accuracy and first-IF activity. The quick mode
  uses the paper's ten activity examples (one per digit).
- `03_srnn.ipynb`: both Braille SRNN variants on the 140 official test
  sequences, including first-hidden-layer activity.

Each notebook generates a canonical **time-driven** NeuroHLS project with
`p=0`, executes only through Vitis HLS CSim, and writes tables, figures, and
provenance to `validation/results/`. CSim establishes functional behaviour;
it does not provide RTL latency, FPGA resources, or power.

The generated validation projects add CSim-only probe output ports for states
that the normal NeuroHLS top keeps internal. These probes are confined to
`validation/build/`; they do not modify the NeuroHLS runtime or claim to be
the public synthesised interface.

## Running

From the repository root, with Vitis 2025.2 available on `PATH`:

```bash
.venv/bin/python validation/run_notebooks.py --experiment all
```

Individual experiments:

```bash
.venv/bin/python validation/run_notebooks.py --experiment lif
.venv/bin/python validation/run_notebooks.py --experiment scnn --scnn-samples 10
.venv/bin/python validation/run_notebooks.py --experiment srnn --srnn-samples 140
```

To render results again without invoking Vitis, reuse the latest successful
runs:

```bash
.venv/bin/python validation/run_notebooks.py --experiment all --no-csim
```

Use `--force` to rebuild projects and disable the CSim cache. The default
arithmetic is floating point so that interoperability differences are not
mixed with quantisation error; `--arithmetic fixed` evaluates the generated
fixed-point contract instead.

## SCNN scope

The paper's published SCNN accuracies use the full 10,000-sample N-MNIST test
set with variable-duration 1 ms frames. The repository-local
`cnn_teste.npz` is a shuffled 1,024-sample subset cropped to 300 steps, while
`cnn_numbers.npy` contains ten examples intended for the activity comparison.
Their NeuroHLS accuracies are therefore labelled with their own sample count
and scope and are **not** presented as a like-for-like replacement for the
published full-test values.

`--scnn-mode local --scnn-samples N` selects the local cropped subset. It can
be slow: an earlier 1,024-sample CSim required roughly 17 hours. The ten-sample
default is intended to make the end-to-end validation practical. To bound log
size, a local-mode probe records hidden activity only for its first ten
samples; that activity is not mixed with the paper's ten digit exemplars.

The small LIF and SRNN reference files are downloaded from the pinned tag when
no official checkout is available. The SCNN activity arrays total roughly
500 MB and are not downloaded by default. Set `NIR_PAPER_REPO` to an existing
checkout, or explicitly pass `--download-large-reference-activity`.

Generated projects, tool runs, and downloaded caches are ignored by Git.
Compact result CSV/JSON/PNG files remain available for inspection.
