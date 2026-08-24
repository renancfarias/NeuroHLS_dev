#!/usr/bin/env python3
"""Generate the time-driven percent-parallel SCNN components of the sweep.

The sweep is indexed by the resolved processing-element count ``U`` rather than
by the requested ``p``: ``U`` is what the generator instantiates, what the
latency law ``lat(U) = W/U + C`` is written in, and what the report and the
dissertation tables label their rows with.  ``p`` is recovered from ``U`` by
inverting ``U(p, W) = round(pW)`` on the dominant dense layer, whose work
domain is ``W = 100352``.

    python3 MLP_test/gerar_componentes_percent_parallelism.py --units 16 32

Regenerating an existing component requires ``--force``, because the projects
are pipeline inputs: overwriting one silently would invalidate the runs already
recorded against its hash under ``sim/runs/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neuro_hls.neuro_hls import NeuroHls  # noqa: E402

COMPONENTS_DIR = REPO_ROOT / "SCNN_test"
MANIFEST = COMPONENTS_DIR / "percent_parallelism_component_manifest.json"

SOURCE_NIR = REPO_ROOT / "nir_examples" / "cnn_sinabs.nir"
METADATA = None   # o grafo Sinabs nao traz metadata de reset
DATASET = COMPONENTS_DIR / "n-mnist-314-steps-medoids.npz"

# Recipe shared by every component of the sweep, so that the only variable
# between two projects is the parallelism.
STEPS_PER_SAMPLE = 314
SAMPLES = 10
BATCH_SIZE = 10
# The dominant dense layer.  U is resolved per layer from each layer's own W,
# so this is the layer whose U names the component.
DOMINANT_WORK_ITEMS = 589824  # Conv2d 2, 64% do total


def requested_p_for_units(units: int, work_items: int = DOMINANT_WORK_ITEMS) -> float:
    """Invert U(p, W) = round(pW) exactly, giving p = U/W.

    That is the same value the generator reports back as ``p_eff``, so the
    request and the effective fraction coincide and the rounding in U(p, W)
    has nothing to absorb.  ``generate`` still asserts the resolved U, since
    the smaller layers round their own U from this same p.
    """
    if units < 1:
        raise ValueError("U must be at least 1")
    return units / work_items


def project_dir(units: int) -> Path:
    return COMPONENTS_DIR / f"hls_time_driven_scnn_w{units:04d}"


def generate(units: int, *, force: bool = False) -> Path:
    folder = project_dir(units)
    if folder.exists():
        if not force:
            raise SystemExit(
                f"{folder} already exists; pass --force to regenerate it"
            )
        shutil.rmtree(folder)

    requested = requested_p_for_units(units)
    hls = NeuroHls(str(folder))
    model = hls.read_nir_file(str(SOURCE_NIR),
                              str(METADATA) if METADATA else None)
    model.define_time_driven_parallelism(requested)
    hls.implement_model(model, use_float=False, backend="time-driven")
    hls.define_test_dataset(
        str(DATASET),
        data_is_binary=True,
        step_count=STEPS_PER_SAMPLE,
        different_sample_per_step=True,
        max_samples=SAMPLES,
    )
    # Each sample is an independent 32-step sequence, so the membrane state
    # must be cleared at its first step; without this the second sample would
    # start from the potentials the first one left behind.
    hls.create_testbench(
        total_samples=SAMPLES, batch_size=BATCH_SIZE, reset_potentials=True
    )

    # Snapshot of the stimulus this project was built against, kept beside the
    # project so a later run can be checked against the exact medoid set even
    # if the dataset file is regenerated.
    medoids = folder / "tb_medoids"
    medoids.mkdir(exist_ok=True)
    for name in ("data.txt", "targets.txt"):
        shutil.copy2(folder / "tb_data" / name, medoids / name)
    shutil.copy2(folder / "testbench.cpp", medoids / "testbench.cpp")

    # A camada dominante nao e a primeira neste grafo -- e a segunda
    # convolucao -- entao ela e localizada pelo dominio de trabalho, e nao
    # pela posicao na lista como na rede feed-forward.
    resolved = json.loads((folder / "parallelism_manifest.json").read_text())
    dominant = max(resolved["layers"], key=lambda l: l["total_work_items"])
    if dominant["processing_elements"] != units:
        raise SystemExit(
            f"requested p={requested!r} resolved to "
            f"U={dominant['processing_elements']} on {dominant['name']}, "
            f"not U={units}"
        )
    return folder


def register(units: int) -> None:
    """Record the component in the sweep manifest, replacing any stale entry."""
    manifest = json.loads(MANIFEST.read_text())
    folder = project_dir(units)
    resolved = json.loads((folder / "parallelism_manifest.json").read_text())
    entry = {
        "name": folder.name,
        "project": str(folder.relative_to(REPO_ROOT)),
        "backend": "time-driven",
        "requested_parallelism": requested_p_for_units(units),
        "plans": [
            {
                "layer": layer["name"],
                "type": layer["operator"],
                "requested_parallelism": layer["requested_parallelism"],
                "total_work_items": layer["total_work_items"],
                "processing_elements": layer["processing_elements"],
                "reuse_cycles": layer["reuse_cycles"],
                "effective_parallelism": layer["effective_parallelism"],
                "idle_slots": layer["idle_slots"],
                "operation_kind": layer["operation_kind"],
            }
            for layer in resolved["layers"]
        ],
        "samples_all_stages": SAMPLES,
        "batch_size_all_stages": BATCH_SIZE,
        "steps_per_sample": STEPS_PER_SAMPLE,
        "dataset": str(DATASET.relative_to(REPO_ROOT)),
    }
    components = [c for c in manifest["components"] if c["name"] != folder.name]
    components.append(entry)
    # The event-driven reference carries no reuse plan, so it has no U to sort
    # on; it belongs at the end, where the report and the charts also put it.
    components.sort(
        key=lambda c: (
            max((pl["processing_elements"] for pl in (c.get("plans") or [])),
                default=float("inf")),
            c["name"],
        )
    )
    manifest["components"] = components
    MANIFEST.write_text(json.dumps(manifest, indent=4) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--units", type=int, nargs="+", required=True,
        help="processing-element counts U to generate",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="regenerate a component whose directory already exists",
    )
    args = parser.parse_args()

    for units in args.units:
        folder = generate(units, force=args.force)
        register(units)
        print(f"U={units:<5d} -> {folder.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
