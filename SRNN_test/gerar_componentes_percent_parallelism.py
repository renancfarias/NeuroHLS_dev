#!/usr/bin/env python3
"""Generate the Braille SRNN percent-parallel sweep and its event-driven reference.

The sweep is indexed by the model-wide request ``p``, not by a per-layer ``U``.
In the feed-forward MLP one dense layer carries 98.6% of the static work, so a
single ``U`` describes the whole design.  Here the dominant layer -- the
recurrent Affine -- carries only 64%, so parallelising it alone would saturate
at 2.8x by Amdahl's law.  A model-wide ``p`` resolves ``U_i = round(p W_i)``
in every layer instead, which drives every reuse count towards ``1/p``.

    python3 SRNN_test/gerar_componentes_percent_parallelism.py --all
    python3 SRNN_test/gerar_componentes_percent_parallelism.py --graph zero --p 0.04

Regenerating an existing component requires ``--force``: the projects are
pipeline inputs, and overwriting one silently would invalidate the runs already
recorded against its hash under ``sim/runs/``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neuro_hls import NeuroHls  # noqa: E402
from neuro_hls.backend_utils import copy_backend_to  # noqa: E402

COMPONENTS_DIR = REPO_ROOT / "SRNN_test"
MANIFEST = COMPONENTS_DIR / "percent_parallelism_component_manifest.json"

DATASET = REPO_ROOT / "samples" / "rnn_test_medoids.pt"
STEPS_PER_SAMPLE = 256
SAMPLES = 7
BATCH_SIZE = 7

# The active-list noise threshold: an event whose contribution falls below it
# is not worth a state update.  Kept identical across the two graphs so the
# event-driven reference differs from its time-driven counterparts only in the
# backend.
ACTIVE_NOISE_THRESHOLD = 1e-6

GRAPHS = {
    "zero": {
        "nir": REPO_ROOT / "nir_examples" / "braille_noDelay_bias_zero.nir",
        "metadata": REPO_ROOT / "nir_examples" / "model_noDelay_bias_ref_zero.metadata.json",
    },
    "subtract": {
        "nir": REPO_ROOT / "nir_examples" / "braille_noDelay_noBias_subtract.nir",
        "metadata": REPO_ROOT / "nir_examples" / "model_noDelay_noBias_ref_subtract.metadata.json",
    },
}

# Ten points doubling from 0.005, plus the serial sentinel and full unrolling.
# Each step roughly halves the summed reuse count, and no two resolve to the
# same architecture -- p=0.7, for instance, is excluded because it produces the
# same plan as p=0.5.
SWEEP_P = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)


def project_name(graph: str, p: float) -> str:
    """Name a component by graph and request, in units of 1e-4 so it sorts."""
    return f"hls_time_driven_{graph}_p{round(p * 10000):05d}"


def event_project_name(graph: str) -> str:
    return f"hls_event_driven_{graph}_active_list"


def _build(folder: Path, graph: str, *, event_driven: bool, p: float = 0.0) -> None:
    spec = GRAPHS[graph]
    copy_backend_to(str(folder))
    hls = NeuroHls(str(folder))
    model = hls.read_nir_file(str(spec["nir"]), metadata_file_path=str(spec["metadata"]))
    if event_driven:
        model.define_event_cuba_lif_strategy("active_list")
        model.define_event_active_noise_threshold(ACTIVE_NOISE_THRESHOLD)
        hls.implement_model(model, use_float=False, backend="event-driven")
    else:
        model.define_time_driven_parallelism(p)
        hls.implement_model(model, use_float=False, backend="time-driven")
    hls.define_test_dataset(
        str(DATASET), data_is_binary=True, step_count=STEPS_PER_SAMPLE,
        different_sample_per_step=True, max_samples=SAMPLES,
    )
    # Each medoid is an independent 256-step sequence, so the membrane and the
    # recurrent state must be cleared at its first step.
    hls.create_testbench(
        total_samples=SAMPLES, batch_size=BATCH_SIZE, reset_potentials=True,
    )


def generate(graph: str, p: float, *, force: bool = False) -> Path:
    folder = COMPONENTS_DIR / project_name(graph, p)
    if folder.exists():
        if not force:
            raise SystemExit(f"{folder} already exists; pass --force to regenerate it")
        shutil.rmtree(folder)
    _build(folder, graph, event_driven=False, p=p)

    plan = json.loads((folder / "parallelism_manifest.json").read_text())
    for layer in plan["layers"]:
        expected = 1 if p == 0.0 else min(
            layer["total_work_items"],
            max(1, math.floor(p * layer["total_work_items"] + 0.5)),
        )
        if layer["processing_elements"] != expected:
            raise SystemExit(
                f"{folder.name}: layer {layer['name']} resolved to "
                f"U={layer['processing_elements']}, expected {expected}"
            )
    return folder


def generate_event(graph: str, *, force: bool = False) -> Path:
    folder = COMPONENTS_DIR / event_project_name(graph)
    if folder.exists():
        if not force:
            raise SystemExit(f"{folder} already exists; pass --force to regenerate it")
        shutil.rmtree(folder)
    _build(folder, graph, event_driven=True)
    return folder


def _entry(graph: str, folder: Path, p: float | None) -> dict:
    entry = {
        "name": folder.name,
        "project": str(folder.relative_to(REPO_ROOT)),
        "graph": graph,
        "backend": "time-driven" if p is not None else "event-driven",
        "requested_parallelism": p,
        "samples_all_stages": SAMPLES,
        "batch_size_all_stages": BATCH_SIZE,
        "steps_per_sample": STEPS_PER_SAMPLE,
        "dataset": str(DATASET.relative_to(REPO_ROOT)),
    }
    if p is None:
        entry["active_noise_threshold"] = ACTIVE_NOISE_THRESHOLD
        return entry
    plan = json.loads((folder / "parallelism_manifest.json").read_text())
    entry["plans"] = [
        {
            "layer": layer["name"],
            "type": layer["operator"],
            "operation_kind": layer["operation_kind"],
            "total_work_items": layer["total_work_items"],
            "processing_elements": layer["processing_elements"],
            "reuse_cycles": layer["reuse_cycles"],
            "effective_parallelism": layer["effective_parallelism"],
            "idle_slots": layer["idle_slots"],
        }
        for layer in plan["layers"]
    ]
    entry["summed_reuse_cycles"] = sum(l["reuse_cycles"] for l in entry["plans"])
    entry["summed_processing_elements"] = sum(
        l["processing_elements"] for l in entry["plans"]
    )
    return entry


def register(graph: str, folder: Path, p: float | None) -> None:
    manifest = (
        json.loads(MANIFEST.read_text())
        if MANIFEST.is_file()
        else {
            "source": "Braille SRNN, two published CUBA-LIF variants",
            "parallelism_contract": "percent_parallel_reuse_v1",
            "note": (
                "p is a model-wide request; each layer resolves its own U from "
                "its own W. HLS must still validate resources, II, timing, and "
                "numerical behaviour."
            ),
            "components": [],
        }
    )
    entry = _entry(graph, folder, p)
    components = [c for c in manifest["components"] if c["name"] != entry["name"]]
    components.append(entry)
    # Event-driven references carry no request; they belong after the sweep.
    components.sort(
        key=lambda c: (
            c["graph"],
            c["requested_parallelism"] is None,
            c["requested_parallelism"] or 0.0,
        )
    )
    manifest["components"] = components
    MANIFEST.write_text(json.dumps(manifest, indent=4) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=sorted(GRAPHS), action="append")
    parser.add_argument("--p", type=float, action="append")
    parser.add_argument("--all", action="store_true", help="both graphs, full sweep, plus references")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    graphs = args.graph or sorted(GRAPHS)
    points = args.p if args.p else (SWEEP_P if args.all else None)
    if points is None:
        parser.error("pass --all or at least one --p")

    COMPONENTS_DIR.mkdir(exist_ok=True)
    for graph in graphs:
        for p in points:
            folder = generate(graph, p, force=args.force)
            register(graph, folder, p)
            plan = json.loads((folder / "parallelism_manifest.json").read_text())
            units = [l["processing_elements"] for l in plan["layers"]]
            reuse = sum(l["reuse_cycles"] for l in plan["layers"])
            print(f"{folder.name:38s} U={units} R={reuse}")
        if args.all or not args.p:
            folder = generate_event(graph, force=args.force)
            register(graph, folder, None)
            print(f"{folder.name:38s} event-driven active-list, eps={ACTIVE_NOISE_THRESHOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
