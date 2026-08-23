"""Shared chart style for the two percent-parallel sweep reports.

Both sweeps answer the same question -- what the parallelism buys and what it
costs -- so they are read side by side and must be drawn the same way.  The
panels are a series against the sweep index rather than a bar per component:
the quantity of interest is the trend, and the event-driven reference is a
single horizontal line the series has to cross.

The x axis is categorical.  The sweeps are geometric but include an exact zero
(the serial sentinel of the recurrent sweep), which no logarithmic axis can
place; spacing the points evenly and labelling them keeps every point visible
and keeps the two reports comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICK_SIZE = 13
AXIS_LABEL_SIZE = 15
PANEL_TITLE_SIZE = 18
LEGEND_SIZE = 13

TIME_DRIVEN_COLOUR = "#1f4fd8"
EVENT_DRIVEN_COLOUR = "#e2582a"
GRID_COLOUR = "#d1d5db"

# (key, panel title, axis unit).  Latency and energy are the outcome; the four
# resources are the price.  A report may draw them together or as two figures,
# whichever matches how its surrounding text reads them.
OUTCOME_PANELS = (
    ("cycles", "Cycles per logical step", "cycles"),
    ("energy_uj", "Energy per logical step", "µJ"),
)
RESOURCE_PANELS = (
    ("lut", "LUT", "LUTs"),
    ("ff", "FF", "flip-flops"),
    ("dsp", "DSP", "DSP slices"),
    ("bram", "BRAM", "block RAM tiles"),
)
PANELS = OUTCOME_PANELS + RESOURCE_PANELS

# URAM is omitted throughout: every component of both sweeps reports zero.
_GRIDS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 6: (2, 3)}


def _log_is_readable(values: Sequence[float]) -> bool:
    """A log axis needs strictly positive values and a range worth expanding.

    BRAM reaches exactly zero in the recurrent sweep, where the partitioning
    stops mapping to block memory, so the check cannot be skipped.
    """
    positive = [v for v in values if v > 0.0]
    if len(positive) != len(values) or len(positive) < 2:
        return False
    return max(positive) / min(positive) >= 20.0


def sweep_figure(
    labels: Sequence[str],
    series: Sequence[dict],
    reference: dict | None,
    *,
    x_label: str,
    reference_label: str = "event-driven reference",
    series_label: str = "time-driven",
    panels: Sequence[tuple[str, str, str]] = PANELS,
) -> plt.Figure:
    """Draw one panel per quantity for one sweep, with the reference as a rule.

    A sweep whose points carry two names -- the feed-forward experiment is
    requested as a fraction ``p`` but reasoned about in the resolved element
    count ``U`` -- puts the second one in ``series_label``, so the axis stays
    single-scaled and the legend carries the correspondence.
    """
    rows, columns = _GRIDS[len(panels)]
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.5 * columns, 4.3 * rows), squeeze=False,
    )
    positions = range(len(labels))

    for axis, (key, title, unit) in zip(axes.flat, panels):
        values = [point[key] for point in series]
        axis.plot(
            positions, values, "o-", color=TIME_DRIVEN_COLOUR,
            linewidth=2, markersize=5, label=series_label,
        )
        if reference is not None and reference.get(key) is not None:
            axis.axhline(
                reference[key], color=EVENT_DRIVEN_COLOUR, linestyle="--",
                linewidth=2, label=reference_label,
            )
        compared = values + (
            [reference[key]] if reference and reference.get(key) is not None else []
        )
        if _log_is_readable(compared):
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)

        axis.set_ylabel(unit, fontsize=AXIS_LABEL_SIZE)
        axis.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
        # Os passos geometricos aproximam os rotulos no inicio da serie do
        # sweep recorrente (0.005 e 0.01 encostam um no outro na horizontal),
        # entao rotulos longos giram.  Os do sweep feed-forward sao um ou dois
        # digitos e cabem deitados, onde giralos so atrapalharia a leitura.
        crowded = any(len(label) > 3 for label in labels)
        axis.set_xticks(
            list(positions), labels, fontsize=TICK_SIZE,
            rotation=45 if crowded else 0,
            ha="right" if crowded else "center",
            rotation_mode="anchor" if crowded else None,
        )
        axis.tick_params(axis="y", labelsize=TICK_SIZE)
        axis.grid(True, which="both", color=GRID_COLOUR, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(title, weight="bold", fontsize=PANEL_TITLE_SIZE)

    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)

    handles, names = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, names, loc="lower center", ncol=len(names),
        fontsize=LEGEND_SIZE, frameon=False,
    )
    # A faixa reservada embaixo mantem a legenda fora dos rotulos do eixo x.
    figure.tight_layout(rect=(0, 0.12 / rows, 1, 1))
    return figure


def save(figure: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    return output
