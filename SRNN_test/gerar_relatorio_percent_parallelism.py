#!/usr/bin/env python3
"""Summarise the Braille SRNN percent-parallel sweep against its event reference.

Indexed by the model-wide request ``p``: the dominant recurrent layer carries
only 64% of the static work, so no single ``U`` describes a design here, unlike
the feed-forward sweep.  The resolved ``U`` of every layer is listed alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "results"))

from estilo_sweep import (  # noqa: E402
    OUTCOME_PANELS, RESOURCE_PANELS, save, sweep_figure,
)

RUNS = REPO_ROOT / "sim" / "runs"
COMPONENTS = REPO_ROOT / "SRNN_test"
REPORT = COMPONENTS / "relatorio_percent_parallelism.md"
CLOCK_HZ = 150e6

SWEEP = [(0.0, "00000"), (0.005, "00050"), (0.01, "00100"), (0.02, "00200"),
         (0.04, "00400"), (0.08, "00800"), (0.16, "01600"), (0.32, "03200"),
         (0.64, "06400"), (1.0, "10000")]
GRAPHS = [("zero", "zero reset, bias"), ("subtract", "subtractive reset, no bias")]

def latest(name: str) -> dict | None:
    runs = sorted((RUNS / name).glob("*/reports/summary.json")) if (RUNS / name).is_dir() else []
    return json.loads(runs[-1].read_text()) if runs else None


def metrics(summary: dict) -> dict | None:
    """Cycles per time step, resources and power, with the latency source.

    Co-simulation is the only source for the event-driven reference: its
    dataflow actors report a data-dependent latency, so synthesis has no
    deterministic figure to give.
    """
    steps = (summary.get("workload") or {}).get("total_logical_steps")
    cosim = summary.get("cosim") or {}
    if cosim.get("total_execution_cycles") and steps:
        cycles, source = cosim["total_execution_cycles"] / steps, "RTL co-sim"
    else:
        worst = ((summary.get("hls") or {}).get("latency") or {}).get("worst_case")
        if not worst:
            return None
        cycles, source = float(worst), "HLS synthesis"
    used = (summary.get("vivado_utilization") or {}).get("resources") or {}
    power = summary.get("power") or {}
    total = power.get("total_on_chip_power_w")
    if not used or total is None:
        return None
    return {
        "cycles": cycles, "source": source,
        "lut": used["lut"]["used"], "ff": used["ff"]["used"],
        "dsp": used["dsp"]["used"], "bram": used["bram"]["used"],
        "total_w": total, "dynamic_w": power.get("dynamic_power_w"),
        "energy_uj": total * cycles / CLOCK_HZ * 1e6,
    }


def plan(name: str) -> list[int] | None:
    manifest = COMPONENTS / name / "parallelism_manifest.json"
    if not manifest.is_file():
        return None
    return [l["processing_elements"] for l in json.loads(manifest.read_text())["layers"]]


def collect(graph: str) -> list[tuple[str, float | None, dict, list[int] | None]]:
    rows = []
    for p, tag in SWEEP:
        name = f"hls_time_driven_{graph}_p{tag}"
        summary = latest(name)
        if summary and (m := metrics(summary)):
            rows.append((name, p, m, plan(name)))
    name = f"hls_event_driven_{graph}_active_list"
    summary = latest(name)
    if summary and (m := metrics(summary)):
        rows.append((name, None, m, None))
    return rows


def label(p: float | None) -> str:
    return "ED active-list" if p is None else f"TD p={p:g}"


def charts(graph: str, rows) -> list[tuple[str, Path]]:
    """Draw the outcome and the price as two figures.

    Latency and energy answer what the parallelism buys; the four resources say
    what it costs.  Keeping them apart lets each be shown next to the paragraph
    that reads it, which is how the feed-forward report is already laid out.
    """
    series = [m for _, p, m, _ in rows if p is not None]
    labels = [("0" if p == 0 else f"{p:g}") for _, p, _, _ in rows if p is not None]
    reference = next((m for _, p, m, _ in rows if p is None), None)

    drawn = []
    for panels, stem, caption in (
        (OUTCOME_PANELS, "tempo_energia", "latency and energy"),
        (RESOURCE_PANELS, "recursos", "resource use"),
    ):
        figure = sweep_figure(
            labels, series, reference,
            x_label="requested p",
            reference_label="event-driven reference",
            panels=panels,
        )
        out = COMPONENTS / f"relatorio_percent_parallelism_{graph}_{stem}.png"
        drawn.append((caption, save(figure, out)))
    return drawn


def main() -> int:
    lines = [
        "# Braille SRNN percent-parallel sweep",
        "",
        "Seven medoid sequences, 256 steps each: 1,792 time steps per design. "
        "Power is a vectorless estimate under default switching activity; every "
        "energy figure is therefore an estimate, not an activity-annotated "
        "measurement.",
        "",
        "`p` is the model-wide request. Each layer resolves its own `U` from its "
        "own work domain `W`, so the `U` column is a vector: "
        "`[fc1, merge, lif1, w_rec, fc2, lif2]`.",
    ]
    for graph, description in GRAPHS:
        rows = collect(graph)
        if not rows:
            continue
        ed = next((m for _, p, m, _ in rows if p is None), None)
        lines += ["", f"## Braille {graph} ({description})", "",
                  "| implementation | U per layer | source | cycles/step | vs. ED | LUT | FF | DSP | BRAM | power (dyn) | energy/step |",
                  "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for name, p, m, units in rows:
            if ed is None or p is None:
                relative = "--"
            elif m["cycles"] < ed["cycles"]:
                relative = f"**{ed['cycles'] / m['cycles']:.2f}x faster**"
            else:
                relative = f"{m['cycles'] / ed['cycles']:.2f}x slower"
            lines.append(
                f"| {label(p)} | {units if units else '--'} | {m['source']} | "
                f"{m['cycles']:,.0f} | {relative} | {m['lut']:,.0f} | {m['ff']:,.0f} | "
                f"{m['dsp']:,.0f} | {m['bram']:.1f} | "
                f"{m['total_w']:.3f} W ({m['dynamic_w']:.3f} W) | "
                f"{m['energy_uj']:.2f} µJ |"
            )
        for caption, out in charts(graph, rows):
            lines += ["", f"![Braille {graph}: {caption}]({out.name})"]

    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
