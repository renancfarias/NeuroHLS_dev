#!/usr/bin/env python3
"""Summarise the percent-parallel time-driven MLP sweep.

The script deliberately reads the per-project ``parallelism_manifest.json``
written by the generator.  This prevents a requested ``p`` from being reported
as though it were necessarily the integer architecture accepted by HLS.  The
event-driven component is listed once as a scalar reference.

Examples
--------
python3 MLP_test/gerar_relatorio_percent_parallelism.py
python3 MLP_test/gerar_relatorio_percent_parallelism.py --output /tmp/report.md
python3 MLP_test/gerar_relatorio_percent_parallelism.py --charts-dir MLP_test/figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep matplotlib's cache out of a potentially read-only home directory when
# this script is executed by the automated experiment environment.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurohls-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from estilo_sweep import (  # noqa: E402
    OUTCOME_PANELS, RESOURCE_PANELS, save as save_sweep_figure, sweep_figure,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENTS_DIR = ROOT / "MLP_test"
DEFAULT_RUNS_ROOT = ROOT / "sim" / "runs"
DEFAULT_REPORT = DEFAULT_COMPONENTS_DIR / "relatorio_percent_parallelism.md"
RESOURCE_ORDER = ("lut", "ff", "bram", "dsp", "uram")
# URAM fica fora do gráfico: nenhuma implementação usa o recurso, e um painel
# constante em zero só consome espaço. A tabela mantém a coluna, que é barata e
# documenta a ausência.
CHART_RESOURCE_ORDER = ("lut", "ff", "bram", "dsp")
# Tamanhos de fonte dos gráficos. As figuras são largas e entram no documento
# escaladas para a largura do texto, o que reduz tudo a cerca de metade; estes
# valores compensam a redução para que os rótulos continuem legíveis impressos.
CHART_ANNOTATION_SIZE = 13
CHART_TICK_SIZE = 13
CHART_AXIS_LABEL_SIZE = 15
CHART_PANEL_TITLE_SIZE = 19
CHART_LEGEND_SIZE = 13
RESOURCE_LABELS = {
    "lut": "LUT",
    "ff": "FF",
    "bram": "BRAM",
    "dsp": "DSP",
    "uram": "URAM",
}
HLS_RESOURCE_KEYS = {
    "lut": ("LUT",),
    "ff": ("FF",),
    "bram": ("BRAM_18K", "BRAM"),
    "dsp": ("DSP",),
    "uram": ("URAM",),
}
BACKEND_COLORS = {
    "time-driven": "#2563eb",
    "event-driven": "#ea580c",
}
STATIC_ENERGY_COLOR = "#94a3b8"


@dataclass(frozen=True)
class ImplementationRecord:
    """One current architecture and its newest available pipeline summary."""

    name: str
    backend: str
    requested_parallelism: float | None
    processing_elements: int | None
    summary_path: Path | None
    summary: dict[str, Any] | None


@dataclass(frozen=True)
class ResourceUsage:
    """A resource value, with the reporting stage that produced it."""

    used: float
    available: float | None
    source: str


@dataclass(frozen=True)
class ExecutionMetrics:
    """Total execution time and energy of one execution window.

    ``duration_seconds`` is absent for a run whose window could not be
    established -- an event-driven design reports data-dependent latency, so
    without a SAIF capture there is nothing to multiply the power by.  Power is
    still reported in that case: it was measured, only the energy was not.
    """

    duration_seconds: Optional[float]
    latency_per_sample_seconds: float | None
    latency_per_step_seconds: float | None
    power_total_w: float | None
    power_dynamic_w: float | None
    power_static_w: float | None
    energy_total_joules: float | None
    energy_dynamic_joules: float | None
    energy_static_joules: float | None
    energy_per_sample_joules: float | None
    energy_per_step_joules: float | None
    executed_samples: int | None
    total_logical_steps: int | None
    provisional: bool
    # Origem da janela: uma energia só é interpretável junto com ela.
    window_source: str | None
    # Presentes apenas onde a co-simulação RTL foi executada.
    latency_min_cycles: int | None
    latency_avg_cycles: int | None
    latency_max_cycles: int | None
    interval_avg_cycles: int | None


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def latest_summary(runs_root: Path, project_name: str) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        (runs_root / project_name).glob("*/reports/summary.json"), reverse=True
    )
    if not candidates:
        return None, None
    path = candidates[0]
    return path, read_json(path)


def as_percentage(value: Any) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.6g}%"


def as_requested_p(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.8g}"


def hls_status(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "no run"
    hls = summary.get("hls") or {}
    if not hls.get("available"):
        return "HLS unavailable"
    latency_block = hls.get("latency") or {}
    latency = (
        hls.get("latency_cycles")
        or hls.get("latency_max_cycles")
        or latency_block.get("worst_case")
        or latency_block.get("max")
    )
    interval = (
        hls.get("interval_cycles")
        or hls.get("interval_max_cycles")
        or (hls.get("interval") or {}).get("worst_case")
    )
    fields = []
    if latency is not None:
        fields.append(f"latency={latency} cycles")
    if interval is not None:
        fields.append(f"II={interval}")
    return ", ".join(fields) if fields else "HLS report available"


def run_status(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "not executed"
    stages = summary.get("stages") or {}
    completed = [
        name for name, stage in stages.items()
        if stage.get("state") in {"success", "skipped"}
    ]
    if not completed:
        return str(summary.get("state", "unknown"))
    return ", ".join(completed)


def time_driven_projects(components_dir: Path) -> list[Path]:
    """Return the sweep projects, minus the explicit serial request.

    ``p=0`` is the serial branch of ``U(p, W)`` and resolves to the same U=1
    design as the smallest non-zero request, down to the generated source.  It
    is kept on disk as a contract check, but reporting it would put two
    identical bars on every chart under the same ``U=1`` label.
    """
    return sorted(
        path
        for path in components_dir.glob("hls_time_driven_percent_*")
        if (path / "parallelism_manifest.json").is_file()
        and requested_parallelism(path) != 0.0
    )


def requested_parallelism(project: Path) -> float | None:
    """Read the model-wide request recorded by the generated manifest."""
    manifest = read_json(project / "parallelism_manifest.json")
    layers = manifest.get("layers") or []
    if not layers:
        return None
    value = layers[0].get("requested_parallelism")
    return None if value is None else float(value)


def processing_elements(project: Path) -> int | None:
    """Read the processing-element count the generator resolved for the model.

    ``p`` is the request; ``U`` is what the generator instantiated, and the two
    are not interchangeable -- the same ``p`` yields a different ``U`` in every
    layer, since ``U`` depends on that layer's work domain ``W``.  Layer 0 is
    the dominant dense layer, so its ``U`` is the one that characterises a run.
    """
    manifest = read_json(project / "parallelism_manifest.json")
    layers = manifest.get("layers") or []
    if not layers:
        return None
    value = layers[0].get("processing_elements")
    return None if value is None else int(value)


def implementation_records(
    components_dir: Path, runs_root: Path
) -> list[ImplementationRecord]:
    """Return current time-driven variants plus one scalar event reference."""
    records = []
    for project in time_driven_projects(components_dir):
        summary_path, summary = latest_summary(runs_root, project.name)
        records.append(
            ImplementationRecord(
                name=project.name,
                backend="time-driven",
                requested_parallelism=requested_parallelism(project),
                processing_elements=processing_elements(project),
                summary_path=summary_path,
                summary=summary,
            )
        )

    event_project = components_dir / "hls_event_driven_scalar"
    if event_project.is_dir():
        summary_path, summary = latest_summary(runs_root, event_project.name)
        records.append(
            ImplementationRecord(
                name=event_project.name,
                backend="event-driven",
                requested_parallelism=None,
                processing_elements=None,
                summary_path=summary_path,
                summary=summary,
            )
        )
    return sorted(
        records,
        key=lambda record: (
            record.backend != "time-driven",
            float("inf")
            if record.processing_elements is None
            else record.processing_elements,
            record.name,
        ),
    )


def synthesised(record: ImplementationRecord) -> bool:
    """True when HLS produced a synthesis result for this component.

    The test is ``hls.available`` and not the presence of a latency figure: a
    dataflow design synthesises perfectly well and still reports its top-level
    latency as ``undef``, because the value depends on the data.
    """
    return bool(((record.summary or {}).get("hls") or {}).get("available"))


def partition_records(
    components_dir: Path, runs_root: Path
) -> tuple[list[ImplementationRecord], list[ImplementationRecord]]:
    """Split the components into those with results and the time-driven ones without.

    Tables and charts must apply the same filter, or a chart would show bars for
    a component the tables deliberately omit.
    """
    todos = implementation_records(components_dir, runs_root)
    sem_resultado = [
        record for record in todos
        if record.backend == "time-driven" and not synthesised(record)
    ]
    restantes = [record for record in todos if record not in sem_resultado]
    return restantes, sem_resultado


def synthesis_status(record: ImplementationRecord) -> str:
    """Why a component was left out: attempted and failed, or never attempted."""
    summary = record.summary
    if summary is None:
        return "not executed"
    state = ((summary.get("stages") or {}).get("hls-synth") or {}).get("state")
    if not state:
        return "synthesis not reached"
    return f"synthesis {state}"


def _as_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def resource_usage(summary: dict[str, Any] | None, resource: str) -> ResourceUsage | None:
    """Prefer post-synthesis Vivado use, falling back to the HLS estimate."""
    if summary is None:
        return None

    vivado_resources = (summary.get("vivado_utilization") or {}).get("resources") or {}
    vivado_entry = vivado_resources.get(resource)
    if isinstance(vivado_entry, dict):
        used = _as_finite_float(vivado_entry.get("used"))
        if used is not None:
            return ResourceUsage(
                used=used,
                available=_as_finite_float(vivado_entry.get("available")),
                source="Vivado OOC",
            )

    hls_resources = (summary.get("hls") or {}).get("resources") or {}
    for key in HLS_RESOURCE_KEYS[resource]:
        used = _as_finite_float(hls_resources.get(key))
        if used is not None:
            return ResourceUsage(used=used, available=None, source="HLS estimate")
    return None


def _as_positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def execution_metrics(summary: dict[str, Any] | None) -> ExecutionMetrics | None:
    """Read the total time and energy the pipeline recorded for a run.

    ``derived_metrics`` is written by ``sim.pipeline.derive_summary_metrics``
    from a single activity window: the total time is the SAIF capture window and
    the total energy is the reported average power times that same window
    (``energy_definition = average_power_times_full_saif_window``).  Recomputing
    those numbers here would risk diverging from the definition the pipeline
    actually applied, so they are read as recorded and a run without them is
    reported as unavailable rather than estimated.
    """
    if summary is None:
        return None
    metrics = summary.get("derived_metrics") or {}
    duration = _as_finite_float(metrics.get("capture_duration_seconds"))
    if duration is not None and duration <= 0.0:
        duration = None
    power = summary.get("power") or {}
    cosim = summary.get("cosim") or {}
    if duration is None and not power and not cosim:
        return None
    workload = summary.get("workload") or {}
    return ExecutionMetrics(
        duration_seconds=duration,
        latency_per_sample_seconds=_as_finite_float(
            metrics.get("average_latency_per_sample_seconds")
        ),
        latency_per_step_seconds=_as_finite_float(
            metrics.get("average_latency_per_step_seconds")
        ),
        power_total_w=_as_finite_float(power.get("total_on_chip_power_w")),
        power_dynamic_w=_as_finite_float(power.get("dynamic_power_w")),
        power_static_w=_as_finite_float(power.get("device_static_power_w")),
        energy_total_joules=_as_finite_float(metrics.get("capture_energy_total_joules")),
        energy_dynamic_joules=_as_finite_float(
            metrics.get("capture_energy_dynamic_joules")
        ),
        energy_static_joules=_as_finite_float(
            metrics.get("capture_energy_static_joules")
        ),
        energy_per_sample_joules=_as_finite_float(
            metrics.get("energy_per_sample_total_joules")
        ),
        energy_per_step_joules=_as_finite_float(
            metrics.get("energy_per_step_total_joules")
        ),
        executed_samples=_as_positive_int(workload.get("executed_samples")),
        total_logical_steps=_as_positive_int(workload.get("total_logical_steps")),
        provisional=bool(metrics.get("power_metrics_provisional", False)),
        window_source=metrics.get("latency_definition"),
        latency_min_cycles=_as_positive_int(cosim.get("latency_min_cycles")),
        latency_avg_cycles=_as_positive_int(cosim.get("latency_avg_cycles")),
        latency_max_cycles=_as_positive_int(cosim.get("latency_max_cycles")),
        interval_avg_cycles=_as_positive_int(cosim.get("interval_avg_cycles")),
    )


def _format_scaled(value: float | None, units: tuple[tuple[float, str], ...]) -> str:
    """Format a physical quantity with the largest unit that keeps it >= 1."""
    if value is None:
        return "N/A"
    if value == 0.0:
        return f"0 {units[0][1]}"
    for scale, unit in units:
        if abs(value) >= scale:
            return f"{value / scale:.4g} {unit}"
    scale, unit = units[-1]
    return f"{value / scale:.3g} {unit}"


def format_seconds(value: float | None) -> str:
    return _format_scaled(
        value, ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns"))
    )


def format_joules(value: float | None) -> str:
    return _format_scaled(
        value, ((1.0, "J"), (1e-3, "mJ"), (1e-6, "uJ"), (1e-9, "nJ"))
    )


def format_watts(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4g} W"


def implementation_label(record: ImplementationRecord) -> str:
    """Name a run by its resolved ``U`` rather than by the requested ``p``.

    The latency law and the result tables are both written in ``U``, so a chart
    labelled by ``p`` cannot be read against them.  ``p`` is also a poor axis on
    its own: this sweep spans values such as 9.96492e-06, which carry no meaning
    to a reader without the accompanying ``W``.
    """
    if record.backend == "event-driven":
        # "scalar" nomeia o contrato de stream do backend -- um evento por
        # transferencia -- e nao uma variante escolhida entre outras: o
        # backend event-driven nao tem paralelismo configuravel.  Como so ha
        # uma implementacao event-driven desta rede, o rotulo generico nao
        # perde informacao e nao sugere um contraste inexistente.
        return "ED\nreference"
    if record.processing_elements is None:
        return "TD\nU=N/A"
    return f"TD\nU={record.processing_elements}"


def _format_resource_value(usage: ResourceUsage) -> str:
    if usage.available is not None and usage.available > 0:
        return f"{usage.used:g} ({100.0 * usage.used / usage.available:.2f}%)"
    return f"{usage.used:g}"


def _chart_point(record: ImplementationRecord) -> dict[str, float | None] | None:
    """One record as the quantities the shared sweep panels expect."""
    latency = cycles_per_logical_step(record)
    metrics = execution_metrics(record.summary)
    if latency is None or metrics is None:
        return None
    joules = metrics.energy_per_step_joules
    point: dict[str, float | None] = {
        "cycles": latency[0],
        "energy_uj": joules * 1e6 if joules is not None else None,
    }
    for resource in ("lut", "ff", "dsp", "bram"):
        usage = resource_usage(record.summary, resource)
        point[resource] = usage.used if usage is not None else None
    return point


def _chart_series(records: list[ImplementationRecord]):
    """Split the sweep from its reference, dropping points missing a quantity.

    A partial point cannot be drawn as a series: the line would join values
    that were not all measured the same way.
    """
    series: list[dict] = []
    labels: list[str] = []
    units: list[int] = []
    reference = None
    for record in records:
        point = _chart_point(record)
        if point is None or any(value is None for value in point.values()):
            continue
        if record.backend == "event-driven":
            reference = point
        else:
            series.append(point)
            labels.append(_requested_label(record.requested_parallelism))
            units.append(record.processing_elements)
    return labels, series, reference, units


def _requested_label(requested: float | None) -> str:
    """Format a requested p for a tick label, two significant figures.

    The sweep spans 1e-5 to 6e-4, so a decimal expansion would be unreadable
    and a bare mantissa would hide the exponent change midway through.
    """
    if not requested:
        return "0"
    exponent = math.floor(math.log10(requested))
    mantissa = requested / 10 ** exponent
    # Arredondar a mantissa pode leva-la a 10 (p=9.965e-06 vira 10.0e-06);
    # renormalizar evita o rotulo fora da forma cientifica.
    if round(mantissa, 1) >= 10.0:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"${mantissa:.1f}{{\\times}}10^{{{exponent}}}$"


def _sweep_chart(records, charts_dir: Path, panels, stem: str) -> Path | None:
    labels, series, reference, units = _chart_series(records)
    if not series:
        return None
    charts_dir.mkdir(parents=True, exist_ok=True)
    # O eixo indexa o sweep pela requisicao, como a metodologia o define; o U
    # resolvido vai para a legenda, porque e nele que a lei de latencia e o
    # texto dos resultados raciocinam.
    figure = sweep_figure(
        labels, series, reference,
        x_label="requested $p$",
        reference_label="event-driven reference",
        series_label="time-driven ($U$ = " + ", ".join(map(str, units)) + ")",
        panels=panels,
    )
    return save_sweep_figure(figure, charts_dir / f"{stem}.png")


def render_resource_chart(
    records: list[ImplementationRecord], charts_dir: Path
) -> Path | None:
    return _sweep_chart(
        records, charts_dir, RESOURCE_PANELS,
        "relatorio_percent_parallelism_recursos",
    )


def render_execution_chart(
    records: list[ImplementationRecord], charts_dir: Path
) -> Path | None:
    return _sweep_chart(
        records, charts_dir, OUTCOME_PANELS,
        "relatorio_percent_parallelism_tempo_energia",
    )


def render_project(project: Path, runs_root: Path) -> tuple[str, list[str]]:
    manifest = read_json(project / "parallelism_manifest.json")
    summary_path, summary = latest_summary(runs_root, project.name)
    layers = manifest.get("layers") or []
    requested = layers[0].get("requested_parallelism") if layers else None
    run_id = summary_path.parents[1].name if summary_path else "--"
    row = "| {name} | {p} | {run_id} | {run} | {hls} |".format(
        name=project.name,
        p=as_requested_p(requested),
        run_id=run_id,
        run=run_status(summary),
        hls=hls_status(summary),
    )
    details = []
    for layer in layers:
        details.append(
            "- `{name}` ({operator}, {kind}): W={work}, U={units}, R={reuse}, "
            "p_eff={effective}, idle={idle}".format(
                name=layer["name"],
                operator=layer["operator"],
                kind=layer["operation_kind"],
                work=layer["total_work_items"],
                units=layer["processing_elements"],
                reuse=layer["reuse_cycles"],
                effective=as_percentage(layer["effective_parallelism"]),
                idle=layer["idle_slots"],
            )
        )
    return row, details


def render_event_reference(components_dir: Path, runs_root: Path) -> str:
    project = components_dir / "hls_event_driven_scalar"
    if not project.is_dir():
        return "| `hls_event_driven_scalar` | -- | -- | not generated | -- |"
    summary_path, summary = latest_summary(runs_root, project.name)
    run_id = summary_path.parents[1].name if summary_path else "--"
    return "| {name} | -- | {run_id} | {run} | one event per stream transfer |".format(
        name=project.name,
        run_id=run_id,
        run=run_status(summary),
    )


def render_resource_table(records: list[ImplementationRecord]) -> list[str]:
    """Return a markdown table with the exact values used by the chart."""
    if not records:
        return ["No current implementation has been generated yet."]

    lines = [
        "| Implementation | reporting source | LUT | FF | BRAM | DSP | URAM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        usages = [resource_usage(record.summary, resource) for resource in RESOURCE_ORDER]
        sources = sorted({usage.source for usage in usages if usage is not None})
        cells = [
            _format_resource_value(usage) if usage is not None else "N/A"
            for usage in usages
        ]
        lines.append(
            "| {label} | {source} | {values} |".format(
                label=implementation_label(record).replace("\n", " "),
                source=" / ".join(sources) if sources else "N/A",
                values=" | ".join(cells),
            )
        )
    return lines


def _with_parenthesised_part(total: str, part: str) -> str:
    """Join a total and its dynamic share, dropping an empty parenthesis."""
    if part == "N/A":
        return total
    return f"{total} ({part})"


WINDOW_SOURCE_LABELS = {
    "full_saif_window_amortized": "SAIF",
    "cosim_total_execution_time": "RTL co-sim",
    "hls_latency_times_logical_steps": "HLS latency",
}


def cycles_per_logical_step(record: ImplementationRecord) -> tuple[float, str] | None:
    """Cycles the implementation spends on one logical step, with its source.

    Two sources, and the distinction matters: co-simulation measures the RTL and
    includes back-pressure stalls, while the synthesis estimate is exact only
    where it reports a deterministic latency.  A design reporting ``undef`` has
    no usable synthesis figure at all, which is why the measured column exists.
    """
    summary = record.summary
    if summary is None:
        return None
    cosim = summary.get("cosim") or {}
    steps = _as_positive_int((summary.get("workload") or {}).get("total_logical_steps"))
    total = _as_finite_float(cosim.get("total_execution_cycles"))
    if total and steps:
        return total / steps, "RTL co-sim"
    latency = (summary.get("hls") or {}).get("latency") or {}
    worst = _as_finite_float(latency.get("worst_case"))
    if worst:
        return worst, "HLS synthesis"
    return None


def render_step_latency_table(records: list[ImplementationRecord]) -> list[str]:
    """Compare cycles per logical step across backends, against the event reference."""
    medidas = []
    for record in records:
        valor = cycles_per_logical_step(record)
        if valor is not None:
            medidas.append((record, valor[0], valor[1]))
    if not medidas:
        return []

    referencia = next(
        (ciclos for record, ciclos, _ in medidas if record.backend == "event-driven"),
        None,
    )
    linhas = []
    for record, ciclos, fonte in sorted(medidas, key=lambda item: item[1]):
        label = implementation_label(record).replace("\n", " ")
        if referencia is None or record.backend == "event-driven":
            relativo = "--"
        elif ciclos < referencia:
            relativo = f"**{referencia / ciclos:.2f}x faster**"
        else:
            relativo = f"{ciclos / referencia:.2f}x slower"
        linhas.append(f"| {label} | {fonte} | {ciclos:,.0f} | {relativo} |")
    return [
        "",
        "### Cycles per logical step",
        "",
        "This is the quantity the parallelism contract is meant to reduce, so it "
        "is compared directly against the event-driven implementation of the same "
        "network. The source column matters: a co-simulated figure is measured on "
        "the RTL and absorbs back-pressure stalls, whereas a synthesis figure is "
        "exact only where the reported latency is deterministic.",
        "",
        "| Implementation | source | cycles / logical step | vs. event-driven |",
        "|---|---|---:|---|",
        *linhas,
    ]


def render_latency_table(records: list[ImplementationRecord]) -> list[str]:
    """Return the measured RTL latency table, or nothing when none was measured.

    Only co-simulation produces these numbers.  A design whose synthesis reports
    a deterministic latency does not need them; one whose latency is
    data-dependent has no other source, because the spread comes from the
    workload and from FIFO back-pressure between dataflow actors.
    """
    rows = []
    for record in records:
        metrics = execution_metrics(record.summary)
        if metrics is None or metrics.latency_avg_cycles is None:
            continue
        label = implementation_label(record).replace("\n", " ")
        spread = (
            f"{metrics.latency_max_cycles / metrics.latency_min_cycles:.0f}x"
            if metrics.latency_min_cycles else "--"
        )
        rows.append(
            "| {label} | {mn} | {avg} | {mx} | {spread} | {interval} |".format(
                label=label,
                mn=f"{metrics.latency_min_cycles:,}" if metrics.latency_min_cycles else "N/A",
                avg=f"{metrics.latency_avg_cycles:,}",
                mx=f"{metrics.latency_max_cycles:,}" if metrics.latency_max_cycles else "N/A",
                spread=spread,
                interval=f"{metrics.interval_avg_cycles:,}" if metrics.interval_avg_cycles else "N/A",
            )
        )
    if not rows:
        return []
    return [
        "",
        "### Measured RTL latency",
        "",
        "These come from C/RTL co-simulation, which is the only source for a "
        "design whose synthesis reports `undef`: the per-transaction latency "
        "varies with the workload, and in a dataflow region it also absorbs the "
        "stalls caused by back-pressure between actors. A trip-count model built "
        "from C simulation captures the first effect but not the second.",
        "",
        "| Implementation | min | avg | max | spread | interval (avg) |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Cycles at the configured clock.",
    ]


def render_execution_table(records: list[ImplementationRecord]) -> list[str]:
    """Return a markdown table with the exact values used by the time/energy chart."""
    if not records:
        return ["No current implementation has been generated yet."]

    lines = [
        "| Implementation | window | total time | time / sample | time / step | "
        "total power (dynamic) | total energy (dynamic) | energy / sample | energy / step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    any_provisional = False
    any_measured = False
    for record in records:
        label = implementation_label(record).replace("\n", " ")
        metrics = execution_metrics(record.summary)
        if metrics is None:
            lines.append(f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        any_measured = True
        any_provisional = any_provisional or metrics.provisional
        lines.append(
            "| {label}{mark} | {window} | {duration} | {per_sample} | {per_step} | "
            "{power} | {energy} | {energy_sample} | {energy_step} |".format(
                label=label,
                mark=" *" if metrics.provisional else "",
                window=WINDOW_SOURCE_LABELS.get(metrics.window_source or "", "N/A"),
                duration=format_seconds(metrics.duration_seconds),
                per_sample=format_seconds(metrics.latency_per_sample_seconds),
                per_step=format_seconds(metrics.latency_per_step_seconds),
                power=_with_parenthesised_part(
                    format_watts(metrics.power_total_w),
                    format_watts(metrics.power_dynamic_w),
                ),
                energy=_with_parenthesised_part(
                    format_joules(metrics.energy_total_joules),
                    format_joules(metrics.energy_dynamic_joules),
                ),
                energy_sample=format_joules(metrics.energy_per_sample_joules),
                energy_step=format_joules(metrics.energy_per_step_joules),
            )
        )
    if not any_measured:
        lines += [
            "",
            "No implementation has completed the post-synthesis activity and "
            "power stages yet, so no time or energy value is available.",
        ]
    if any_provisional:
        lines += [
            "",
            "`*` marks a run the pipeline flagged as `power_metrics_provisional`. "
            "That happens either because power was estimated vectorless, from "
            "default toggle rates rather than the workload's own activity, or "
            "because a SAIF capture was partial. The run's `activity_source` "
            "distinguishes the two.",
        ]
    return lines


def build_report(
    components_dir: Path,
    runs_root: Path,
    resource_chart_reference: str | None = None,
    execution_chart_reference: str | None = None,
) -> str:
    projects = time_driven_projects(components_dir)
    lines = [
        "# Percent-parallel reuse report",
        "",
        "This report applies only to the current time-driven percent-parallel "
        "reuse contract. Legacy directories containing `reduction` and "
        "event-driven variants named by `p` are historical artefacts and are "
        "not included.",
        "",
        "| Component | requested p | latest run | completed stages | HLS status |",
        "|---|---:|---|---|---|",
    ]
    pending_details: list[tuple[str, list[str]]] = []
    for project in projects:
        row, details = render_project(project, runs_root)
        lines.append(row)
        pending_details.append((project.name, details))
    lines.append(render_event_reference(components_dir, runs_root))

    # Só o time-driven é filtrado: o componente event-driven permanece nas
    # tabelas de qualquer forma, por ser a referência de comparação.
    records, sem_resultado = partition_records(components_dir, runs_root)
    lines += [
        "",
        "## Resource use",
        "",
        "The chart and table prefer post-synthesis Vivado out-of-context "
        "utilisation. If that stage is unavailable, a hatched bar and the "
        "table source identify an HLS estimate rather than a post-synthesis result.",
        "",
        "In the chart below, `TD` marks the time-driven implementations and `ED` "
        "the event-driven one, which is the reference they are compared against. "
        "A resource with no value for a given implementation is annotated `N/A` "
        "on its axis. The chart has one panel per resource that some "
        "implementation actually uses; URAM is omitted because every component "
        "reports zero, and the table below keeps the column so that the absence "
        "is on record.",
        "",
        *render_resource_table(records),
    ]
    if resource_chart_reference is not None:
        lines += [
            "",
            f"![Resource use by implementation]({resource_chart_reference})",
        ]
    else:
        lines += [
            "",
            "No HLS or Vivado resource report is available for the current "
            "implementations, so no resource chart was generated.",
        ]

    lines += [
        "",
        "## Execution time and energy",
        "",
        "Both quantities come from a single execution window, and the total "
        "energy is the Vivado average power for that window multiplied by its "
        "duration. The `window` column names the origin, because an energy "
        "figure is only interpretable together with it. Three origins occur, in "
        "decreasing order of directness: the SAIF capture measured by the "
        "gate-level simulation; the total execution time measured by C/RTL "
        "co-simulation; and the HLS latency times the clock period times the "
        "logical step count, which is available only where synthesis reports a "
        "deterministic latency. Values are read from the run's "
        "`derived_metrics`, not recomputed here.",
        "",
        "Static and dynamic energy are reported separately because only the "
        "dynamic share follows the design: device static power is a property of "
        "the part and is charged for the whole window regardless of `p`.",
        "",
        "The chart at the end of this section shows the same two quantities. "
        "`TD` marks the time-driven implementations and `ED` the event-driven "
        "one; a `*` marks a provisional value, under the same rule as the table "
        "below. The energy bar is split into its static and dynamic parts, the "
        "dynamic segment coloured by backend, and either axis switches to a "
        "logarithmic scale when the implementations span more than a factor of "
        "fifty, which would otherwise leave the fastest bar invisible.",
        "",
        *render_execution_table(records),
        *render_step_latency_table(records),
        *render_latency_table(records),
    ]
    if execution_chart_reference is not None:
        lines += [
            "",
            f"![Execution time and energy by implementation]({execution_chart_reference})",
        ]

    if not projects:
        lines += [
            "",
            "No percent-parallel time-driven project is available yet. Generate "
            "the components with the updated MLP notebook first.",
        ]
        return "\n".join(lines) + "\n"

    lines += ["", "## Resolved time-driven plans", ""]
    for name, details in pending_details:
        lines += [f"### `{name}`", *details, ""]
    lines += [
        *(
            [
                "",
                "## Time-driven components without results",
                "",
                "These produced no synthesis result, so they carry no resource, "
                "timing or energy figure and are omitted from the tables above. "
                "The status separates a component whose synthesis was attempted "
                "and failed from one that was never run: only the first is "
                "evidence that the configuration is not synthesisable.",
                "",
                "| Component | requested p | status |",
                "|---|---:|---|",
                *(
                    "| `{name}` | {p} | {status} |".format(
                        name=record.name,
                        p=as_requested_p(record.requested_parallelism),
                        status=synthesis_status(record),
                    )
                    for record in sem_resultado
                ),
            ]
            if sem_resultado
            else []
        ),
        "",
        "## Interpretation",
        "",
        "`p` is only the requested percentage. Compare runs using the resolved "
        "`W`, `U`, `R`, and `p_eff` values above, then use the HLS/co-simulation "
        "reports for achieved II and latency. Resource, timing, and power data "
        "must not be inferred from `R` alone.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-dir", type=Path, default=DEFAULT_COMPONENTS_DIR)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--charts-dir",
        type=Path,
        help="directory for PNG/SVG charts (default: report directory)",
    )
    arguments = parser.parse_args()

    charts_dir = arguments.charts_dir or arguments.output.parent
    records, _ = partition_records(arguments.components_dir, arguments.runs_root)

    def chart_reference(chart_path: Path | None) -> str | None:
        if chart_path is None:
            return None
        return Path(
            os.path.relpath(chart_path, arguments.output.parent)
        ).as_posix()

    resource_reference = chart_reference(render_resource_chart(records, charts_dir))
    execution_reference = chart_reference(render_execution_chart(records, charts_dir))

    arguments.output.write_text(
        build_report(
            arguments.components_dir,
            arguments.runs_root,
            resource_chart_reference=resource_reference,
            execution_chart_reference=execution_reference,
        ),
        encoding="utf-8",
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
