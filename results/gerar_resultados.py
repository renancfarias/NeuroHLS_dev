#!/usr/bin/env python3
"""Compila os summaries mais recentes e gera tabelas, CSV e gráficos.

O script descobre automaticamente as variantes `event_driven_*` e
`time_driven_*` em `sim/runs`. Para cada variante ele escolhe primeiro o
diretório de run com o timestamp UTC mais recente e só então procura o
`reports/summary.md`; nunca há fallback silencioso para um run antigo.

Saídas padrão:

* results/comparativo_event_time_driven.md
* results/metricas_compiladas.csv
* results/graficos/01_etapas.svg ... 09_qualidade_saif.svg
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import shlex
import tempfile
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


NETWORK_RE = re.compile(r"^(event|time)_driven(?:_.+)$")
RUN_RE = re.compile(r"^(\d{8}T\d{6}Z)-(.+)$")
NUMBER_RE = re.compile(r"[-+]?[0-9][0-9.,]*")
CSIM_ACCURACY_RE = re.compile(
    r"^\s*\*{3}\s*Final\s+Acc:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

STAGE_ORDER = [
    "prepare",
    "vitis-project",
    "csim",
    "hls-synth",
    "cosim-setup",
    "export",
    "vivado-synth",
    "post-synth-sim",
    "power",
]
HLS_RESOURCES = ["LUT", "FF", "BRAM_18K", "DSP", "URAM"]
VIVADO_RESOURCES = ["LUT", "FF", "BRAM", "DSP", "URAM"]

STATE_REPORTED = "reportado"
STATE_PROVISIONAL = "provisório"
STATE_LOG = "log"
STATE_UNDEFINED = "undef"
STATE_MISSING = "N/D"


@dataclass(frozen=True)
class Metric:
    value: int | Decimal | str | None
    raw: str | None = None
    state: str = STATE_REPORTED
    source: Path | None = None


@dataclass(frozen=True)
class MetricSpec:
    key: str
    group: str
    label: str
    unit: str = ""
    decimals: int | None = None
    kind: str = "number"


@dataclass
class RunResult:
    network: str
    backend: str
    run_id: str
    run_dir: Path
    summary_path: Path
    summary_exists: bool
    project: str | None = None
    top: str | None = None
    fpga: str | None = None
    clock_mhz: Decimal | None = None
    clock_ns: Decimal | None = None
    scope_from_stage: str | None = None
    scope_to_stage: str | None = None
    stages: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    capture_is_partial: bool = False
    capture_transaction_limit: int | None = None

    @property
    def required_stages(self) -> list[str]:
        if self.scope_to_stage in STAGE_ORDER:
            return STAGE_ORDER[: STAGE_ORDER.index(self.scope_to_stage) + 1]
        return STAGE_ORDER

    def includes_stage(self, stage: str) -> bool:
        return stage in self.required_stages

    @property
    def incomplete(self) -> bool:
        if not self.summary_exists:
            return True
        return any(
            normalize(self.stages.get(stage, "")) not in {"success", "skipped"}
            for stage in self.required_stages
        )

    @property
    def power_provisional(self) -> bool:
        return any(
            metric.state == STATE_PROVISIONAL
            for key, metric in self.metrics.items()
            if key.startswith(("power.", "energy."))
        )


HLS_SPECS = [
    MetricSpec(f"hls.{resource}", "HLS", resource)
    for resource in HLS_RESOURCES
]
HLS_LATENCY_SPECS = [
    MetricSpec("hls.latency.best", "HLS", "Latência HLS — melhor caso", kind="text"),
    MetricSpec("hls.latency.average", "HLS", "Latência HLS — média", kind="text"),
    MetricSpec("hls.latency.worst", "HLS", "Latência HLS — pior caso", kind="text"),
]
VIVADO_USED_SPECS = [
    MetricSpec(
        f"vivado.{resource}.used",
        "Vivado OOC",
        resource,
        decimals=1 if resource == "BRAM" else 0,
    )
    for resource in VIVADO_RESOURCES
]
WORKLOAD_SPECS = [
    MetricSpec("workload.samples", "Carga", "Amostras"),
    MetricSpec("workload.steps_per_sample", "Carga", "Passos temporais por amostra"),
    MetricSpec("workload.steps_total", "Carga", "Passos temporais executados"),
    MetricSpec("workload.batches", "Carga", "Batches"),
    MetricSpec("workload.batch_size", "Carga", "Tamanho do batch"),
]
CSIM_SPECS = [
    MetricSpec("csim.accuracy_pct", "CSim", "Acurácia final", "%", 2),
]
SAIF_SPECS = [
    MetricSpec("saif.duration_raw", "SAIF", "Duração bruta SAIF", decimals=1),
    MetricSpec("saif.transitions", "SAIF", "Transições"),
]
TIMING_SPECS = [
    MetricSpec("timing.duration_ms", "Duração e latência", "Duração total", "ms", 9),
    MetricSpec("timing.duration_ps", "Duração e latência", "Duração usada no cálculo", "ps", 0),
    MetricSpec(
        "timing.latency_step_us",
        "Duração e latência",
        "Latência média amortizada",
        "µs/step",
        6,
    ),
    MetricSpec(
        "timing.latency_sample_us",
        "Duração e latência",
        "Latência média amortizada",
        "µs/amostra",
        6,
    ),
]
POWER_SPECS = [
    MetricSpec("power.total_w", "Potência", "Potência total", "W", 3),
    MetricSpec("power.dynamic_w", "Potência", "Potência dinâmica", "W", 3),
    MetricSpec("power.static_w", "Potência", "Potência estática", "W", 3),
]
ENERGY_SPECS = [
    MetricSpec("energy.total_mj", "Energia", "Energia total da simulação", "mJ", 6),
    MetricSpec(
        "energy.step_uj",
        "Energia",
        "Energia média por passo temporal",
        "µJ/step",
        6,
    ),
    MetricSpec(
        "energy.sample_mj",
        "Energia",
        "Energia média por amostra",
        "mJ/amostra",
        6,
    ),
]
ENERGY_CALC_SPECS = [
    MetricSpec("energy.calc_total_mj", "Energia calculada", "E_total", "mJ", 9),
    MetricSpec("energy.calc_step_uj", "Energia calculada", "E_step", "µJ", 9),
    MetricSpec("energy.calc_sample_mj", "Energia calculada", "E_amostra", "mJ", 9),
]
QUALITY_SPECS = [
    MetricSpec("quality.confidence", "Qualidade SAIF", "Confiança geral", kind="text"),
    MetricSpec("quality.annotated_nets", "Qualidade SAIF", "Nets anotados"),
    MetricSpec("quality.total_nets", "Qualidade SAIF", "Nets totais"),
    MetricSpec("quality.coverage_pct", "Qualidade SAIF", "Cobertura reportada", "%"),
    MetricSpec("quality.threshold_pct", "Qualidade SAIF", "Limite de cobertura", "%"),
]
EXTRA_SPECS = [
    MetricSpec("saif.file", "SAIF", "Arquivo SAIF", kind="text"),
    MetricSpec("saif.strip_path", "SAIF", "Strip path", kind="text"),
]
ALL_SPECS = [
    *HLS_SPECS,
    *HLS_LATENCY_SPECS,
    *[
        MetricSpec(
            f"vivado.{resource}.{field_name}",
            "Vivado OOC",
            f"{resource} — {label}",
            "%" if field_name == "usage_pct" else "",
            (
                3
                if field_name == "usage_pct"
                else 1
                if field_name == "used" and resource == "BRAM"
                else 0
            ),
        )
        for resource in VIVADO_RESOURCES
        for field_name, label in [
            ("used", "utilizado"),
            ("available", "disponível"),
            ("usage_pct", "uso"),
        ]
    ],
    *WORKLOAD_SPECS,
    *CSIM_SPECS,
    *SAIF_SPECS,
    *EXTRA_SPECS,
    *TIMING_SPECS,
    *POWER_SPECS,
    *ENERGY_SPECS,
    *ENERGY_CALC_SPECS,
    *QUALITY_SPECS,
]


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#B8B8B8",
        "xtick.color": "#444444",
        "ytick.color": "#222222",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "neurohls-driven-results",
    }
)


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    unaccented = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    unaccented = unaccented.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", unaccented).strip().lower()


def parse_raw_decimal(raw: str) -> Decimal:
    return Decimal(raw.strip())


def parse_pt_decimal(raw: str) -> Decimal:
    cleaned = raw.strip().replace(".", "").replace(",", ".")
    return Decimal(cleaned)


def parse_pt_integer(raw: str) -> int:
    value = parse_pt_decimal(raw)
    if value != value.to_integral_value():
        raise ValueError(f"Esperado inteiro, recebido {raw!r}")
    return int(value)


def numeric_token(text: str) -> str:
    match = NUMBER_RE.search(text)
    if not match:
        raise ValueError(f"Número não encontrado em {text!r}")
    return match.group(0)


def code_spans(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def first_code(text: str) -> str | None:
    spans = code_spans(text)
    return spans[0] if spans else None


def split_sections(text: str) -> tuple[str, dict[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    preamble = text[: matches[0].start()] if matches else text
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[normalize(match.group(1))] = text[match.end() : end]
    return preamble, sections


def section_with_prefix(sections: dict[str, str], prefix: str) -> str | None:
    normalized_prefix = normalize(prefix)
    for heading, content in sections.items():
        if heading.startswith(normalized_prefix):
            return content
    return None


def parse_bullets(text: str) -> dict[str, str]:
    bullets: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^-\s+(?:`([^`]+)`|([^:]+)):\s*(.+?)\s*$", line)
        if match:
            label = match.group(1) or match.group(2)
            bullets[normalize(label)] = match.group(3)
    return bullets


def parse_table(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        if normalize(cells[0]) in {"metrica", "recurso"}:
            continue
        rows[normalize(cells[0])] = cells[1:]
    return rows


def put(
    result: RunResult,
    key: str,
    value: int | Decimal | str | None,
    raw: str | None = None,
    state: str = STATE_REPORTED,
    source: Path | None = None,
) -> None:
    result.metrics[key] = Metric(
        value=value,
        raw=raw,
        state=state,
        source=source,
    )


def parse_number_cell(
    result: RunResult,
    key: str,
    cell: str,
    parser,
    state: str = STATE_REPORTED,
    source: Path | None = None,
) -> None:
    raw_code = first_code(cell)
    if raw_code is None:
        result.warnings.append(f"{key}: valor entre crases não encontrado")
        return
    if normalize(raw_code) in {"n/d", "nd", "n/a", "na", "-"}:
        put(result, key, None, raw_code, STATE_MISSING, source)
        return
    try:
        put(
            result,
            key,
            parser(numeric_token(raw_code)),
            raw_code,
            state,
            source,
        )
    except (InvalidOperation, ValueError) as error:
        result.warnings.append(f"{key}: {error}")


def parse_summary(selection: RunResult) -> RunResult:
    result = selection
    if not result.summary_exists:
        result.warnings.append(
            "O run mais recente não contém reports/summary.md; métricas mantidas como N/D."
        )
        return result

    text = result.summary_path.read_text(encoding="utf-8")
    partial_capture = re.search(
        r"Captura parcial:\s*limitada a `([0-9]+)` transações\s+por",
        text,
        flags=re.IGNORECASE,
    )
    if partial_capture:
        result.capture_is_partial = True
        result.capture_transaction_limit = int(partial_capture.group(1))
    preamble, sections = split_sections(text)
    metadata = parse_bullets(preamble)

    def metadata_code(label: str) -> str | None:
        value = metadata.get(normalize(label))
        return first_code(value) if value else None

    reported_run = metadata_code("Run")
    if reported_run and reported_run != result.run_id:
        result.warnings.append(
            f"Run do cabeçalho ({reported_run}) difere do diretório ({result.run_id})."
        )
    result.project = metadata_code("Projeto")
    result.top = metadata_code("Top")
    result.fpga = metadata_code("FPGA")
    clock_text = metadata.get(normalize("Clock"), "")
    clock_values = code_spans(clock_text)
    try:
        if clock_values:
            result.clock_mhz = parse_raw_decimal(numeric_token(clock_values[0]))
        if len(clock_values) > 1:
            result.clock_ns = parse_raw_decimal(numeric_token(clock_values[1]))
    except (InvalidOperation, ValueError) as error:
        result.warnings.append(f"Clock: {error}")

    stages_section = section_with_prefix(sections, "Etapas")
    if stages_section:
        stage_bullets = parse_bullets(stages_section)
        result.stages = {
            stage: first_code(value) or value.strip()
            for stage, value in stage_bullets.items()
        }

    scope_section = section_with_prefix(sections, "Escopo da execução")
    if scope_section:
        scope = parse_bullets(scope_section)
        from_stage = first_code(scope.get(normalize("Etapa inicial"), ""))
        to_stage = first_code(scope.get(normalize("Etapa final"), ""))
        if from_stage in STAGE_ORDER:
            result.scope_from_stage = from_stage
        if to_stage in STAGE_ORDER:
            result.scope_to_stage = to_stage

    csim_section = section_with_prefix(sections, "Acurácia CSim")
    if csim_section:
        csim = parse_bullets(csim_section)
        accuracy_cell = csim.get(normalize("Acurácia final"), "")
        log_raw = first_code(csim.get(normalize("Log"), ""))
        log_path: Path | None = None
        if log_raw:
            log_path = Path(log_raw)
            if not log_path.is_absolute():
                log_path = (result.run_dir / log_path).resolve()
        if accuracy_cell:
            parse_number_cell(
                result,
                "csim.accuracy_pct",
                accuracy_cell,
                parse_pt_decimal,
                source=log_path,
            )

    saif_section = section_with_prefix(sections, "Atividade SAIF")
    if saif_section:
        saif = parse_bullets(saif_section)
        file_raw = first_code(saif.get(normalize("Arquivo"), ""))
        duration_raw = first_code(saif.get(normalize("Duração"), ""))
        transitions_raw = first_code(saif.get(normalize("Transições"), ""))
        strip_raw = first_code(saif.get(normalize("Strip path"), ""))
        if file_raw:
            file_path = Path(file_raw)
            if not file_path.is_absolute():
                file_path = (result.run_dir / file_path).resolve()
            put(result, "saif.file", file_path.as_posix(), file_raw)
        if strip_raw:
            put(result, "saif.strip_path", strip_raw, strip_raw)
        try:
            if duration_raw:
                put(
                    result,
                    "saif.duration_raw",
                    parse_raw_decimal(duration_raw),
                    duration_raw,
                )
            if transitions_raw:
                put(
                    result,
                    "saif.transitions",
                    int(parse_raw_decimal(transitions_raw)),
                    transitions_raw,
                )
        except (InvalidOperation, ValueError) as error:
            result.warnings.append(f"Atividade SAIF: {error}")

    hls_section = section_with_prefix(sections, "Estimativa de recursos HLS")
    if hls_section:
        hls_rows = parse_table(hls_section)
        hls_log_source = (
            resource_log_path(
                result.run_dir,
                "project/vitis_proj/sol/syn/report/csynth.rpt",
                "csynth.rpt",
            )
            if "csynth.rpt" in normalize(hls_section)
            else None
        )
        hls_state = STATE_LOG if hls_log_source else STATE_REPORTED
        for resource in HLS_RESOURCES:
            cells = hls_rows.get(normalize(resource))
            if cells:
                parse_number_cell(
                    result,
                    f"hls.{resource}",
                    cells[0],
                    parse_pt_integer,
                    hls_state,
                    hls_log_source,
                )
        latency_match = re.search(
            r"Latência HLS:\s*melhor caso `([^`]+)`,\s*"
            r"média `([^`]+)` e pior caso `([^`]+)`",
            hls_section,
            re.DOTALL,
        )
        if latency_match:
            for key, raw in zip(
                [
                    "hls.latency.best",
                    "hls.latency.average",
                    "hls.latency.worst",
                ],
                latency_match.groups(),
            ):
                normalized_raw = normalize(raw)
                if normalized_raw == "undef":
                    put(result, key, None, raw, STATE_UNDEFINED)
                elif normalized_raw in {"n/d", "nd", "n/a", "na"}:
                    put(result, key, None, raw, STATE_MISSING)
                else:
                    put(result, key, raw, raw)

    vivado_section = section_with_prefix(
        sections,
        "Uso de recursos pós-síntese",
    )
    if vivado_section:
        vivado_rows = parse_table(vivado_section)
        vivado_log_source = (
            resource_log_path(
                result.run_dir,
                "40_vivado_synth/utilization_post_synth.rpt",
                "utilization_post_synth.rpt",
            )
            if "utilization_post_synth.rpt" in normalize(vivado_section)
            else None
        )
        vivado_state = STATE_LOG if vivado_log_source else STATE_REPORTED
        for resource in VIVADO_RESOURCES:
            cells = vivado_rows.get(normalize(resource))
            if not cells or len(cells) < 3:
                continue
            parsers = [
                parse_pt_decimal,
                parse_pt_decimal,
                parse_pt_decimal,
            ]
            for field_name, cell, parser in zip(
                ["used", "available", "usage_pct"],
                cells[:3],
                parsers,
            ):
                parse_number_cell(
                    result,
                    f"vivado.{resource}.{field_name}",
                    cell,
                    parser,
                    vivado_state,
                    vivado_log_source,
                )

    workload_section = section_with_prefix(sections, "Carga da simulação")
    if workload_section:
        workload = parse_bullets(workload_section)
        workload_fields = {
            "workload.samples": "Amostras",
            "workload.steps_per_sample": "Passos temporais por amostra",
            "workload.steps_total": "Passos temporais executados",
            "workload.batches": "Batches",
            "workload.batch_size": "Tamanho do batch",
        }
        for key, label in workload_fields.items():
            raw = first_code(workload.get(normalize(label), ""))
            if not raw:
                continue
            try:
                put(result, key, parse_pt_integer(numeric_token(raw)), raw)
            except (InvalidOperation, ValueError) as error:
                result.warnings.append(f"{key}: {error}")

    timing_section = section_with_prefix(sections, "Duração e latência")
    if timing_section:
        timing_rows = parse_table(timing_section)
        timing_fields = {
            normalize("Duração simulada total (tempo lógico)"): (
                "timing.duration_ms",
                parse_pt_decimal,
            ),
            normalize("Latência média amortizada por passo temporal"): (
                "timing.latency_step_us",
                parse_pt_decimal,
            ),
            normalize("Latência média amortizada por amostra"): (
                "timing.latency_sample_us",
                parse_pt_decimal,
            ),
        }
        for label, (key, parser) in timing_fields.items():
            cells = timing_rows.get(label)
            if cells:
                parse_number_cell(result, key, cells[0], parser)
        duration_cells = timing_rows.get(
            normalize("Duração simulada total (tempo lógico)")
        )
        if duration_cells and len(duration_cells) > 1:
            parse_number_cell(
                result,
                "timing.duration_ps",
                duration_cells[1],
                parse_pt_integer,
            )

    power_section = section_with_prefix(sections, "Potência e energia")
    if power_section:
        power_rows = parse_table(power_section)
        power_fields = {
            "power.total_w": "Potência total",
            "power.dynamic_w": "Potência dinâmica",
            "power.static_w": "Potência estática",
            "energy.total_mj": "Energia total da simulação",
            "energy.step_uj": "Energia média por passo temporal",
            "energy.sample_mj": "Energia média por amostra",
        }
        for key, label in power_fields.items():
            cells = power_rows.get(normalize(label))
            if cells:
                parse_number_cell(result, key, cells[0], parse_pt_decimal)

        exact_patterns = {
            "energy.calc_total_mj": r"E_total\s*=.*?=\s*([0-9.,]+)\s*mJ",
            "energy.calc_step_uj": r"E_step\s*=.*?=\s*([0-9.,]+)\s*µJ",
            "energy.calc_sample_mj": r"E_amostra\s*=.*?=\s*([0-9.,]+)\s*mJ",
        }
        for key, pattern in exact_patterns.items():
            match = re.search(pattern, power_section)
            if match:
                raw = match.group(1)
                try:
                    put(result, key, parse_pt_decimal(raw), raw)
                except InvalidOperation as error:
                    result.warnings.append(f"{key}: {error}")

        collapsed = re.sub(r"\s+", " ", power_section)
        quality = re.search(
            r"confiança geral `([^`]+)`, com ([0-9.,]+)% dos nets "
            r"anotados pelo SAIF"
            r"(?: \(`([0-9.]+)/([0-9.]+)`\))?, "
            r"para um limite de ([0-9.,]+)%",
            collapsed,
        )
        if quality:
            put(result, "quality.confidence", quality.group(1), quality.group(1))
            put(
                result,
                "quality.coverage_pct",
                parse_pt_decimal(quality.group(2)),
                quality.group(2),
            )
            if quality.group(3) is not None and quality.group(4) is not None:
                put(
                    result,
                    "quality.annotated_nets",
                    parse_pt_integer(quality.group(3)),
                    quality.group(3),
                )
                put(
                    result,
                    "quality.total_nets",
                    parse_pt_integer(quality.group(4)),
                    quality.group(4),
                )
            put(
                result,
                "quality.threshold_pct",
                parse_pt_decimal(quality.group(5)),
                quality.group(5),
            )
        elif "vivado informou confiança geral" in normalize(collapsed):
            result.warnings.append(
                "Qualidade SAIF: frase encontrada, mas o formato não pôde ser analisado."
            )

    power_status = normalize(result.stages.get("power", ""))
    has_power = any(
        key.startswith(("power.", "energy."))
        for key in result.metrics
    )
    provisional_text = "provisori" in normalize(text)
    if has_power and (power_status != "success" or provisional_text):
        for key, metric in list(result.metrics.items()):
            if key.startswith(("power.", "energy.")):
                result.metrics[key] = Metric(
                    value=metric.value,
                    raw=metric.raw,
                    state=STATE_PROVISIONAL,
                    source=metric.source,
                )

    return result


def supplement_csim_accuracy_from_log(result: RunResult) -> None:
    """Preenche a acurácia somente com o ``csim.log`` do run selecionado."""
    current = result.metrics.get("csim.accuracy_pct")
    if current is not None and current.state != STATE_MISSING:
        return
    log_path = result.run_dir / "logs" / "csim.log"
    if not log_path.is_file():
        return
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    matches = CSIM_ACCURACY_RE.findall(contents)
    if not matches:
        return
    try:
        accuracy = Decimal(matches[-1])
    except InvalidOperation:
        return
    if not Decimal(0) <= accuracy <= Decimal(100):
        return
    put(
        result,
        "csim.accuracy_pct",
        accuracy,
        matches[-1],
        STATE_LOG,
        log_path,
    )


def resource_log_path(run_dir: Path, relative: str, filename: str) -> Path | None:
    """Retorna o relatório de recursos preferencial do run.

    Os caminhos conhecidos são usados primeiro. O fallback recursivo cobre
    pequenas mudanças de layout sem atravessar o diretório de outro run.
    """
    preferred = run_dir / relative
    if preferred.is_file():
        return preferred
    candidates = sorted(
        path for path in run_dir.rglob(filename) if path.is_file()
    )
    return candidates[0] if candidates else None


def log_table_cells(line: str) -> list[str]:
    if not line.lstrip().startswith("|"):
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def log_integer(cell: str) -> int | None:
    match = re.search(r"(?<![A-Za-z])[0-9][0-9,]*", cell)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def first_log_row(
    report_path: Path,
    predicate,
) -> list[str] | None:
    lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        cells = log_table_cells(line)
        if cells and predicate(cells):
            return cells
    return None


def supplement_resources_from_logs(result: RunResult) -> None:
    """Preenche lacunas de recursos com os relatórios do run selecionado.

    O summary continua sendo a fonte prioritária. Os valores suplementares são
    marcados com ``STATE_LOG`` e retêm o caminho do relatório para auditoria.
    Valores ``-`` dos relatórios HLS permanecem N/D, pois não representam zero.
    """
    hls_keys = [f"hls.{resource}" for resource in HLS_RESOURCES]
    if any(
        result.metrics.get(key) is None
        or result.metrics[key].state == STATE_MISSING
        for key in hls_keys
    ):
        hls_path = resource_log_path(
            result.run_dir,
            "project/vitis_proj/sol/syn/report/csynth.rpt",
            "csynth.rpt",
        )
        if hls_path:
            row = first_log_row(
                hls_path,
                lambda cells: normalize(cells[0].lstrip("+ ")) == "snn_to_hls",
            )
            if row is None or len(row) <= 14:
                result.warnings.append(
                    f"Não foi possível localizar a linha de recursos HLS em {hls_path}."
                )
            else:
                hls_columns = {
                    "hls.BRAM_18K": 10,
                    "hls.DSP": 11,
                    "hls.FF": 12,
                    "hls.LUT": 13,
                    "hls.URAM": 14,
                }
                for key, index in hls_columns.items():
                    if result.metrics.get(key) is not None:
                        continue
                    value = log_integer(row[index])
                    if value is not None:
                        put(
                            result,
                            key,
                            value,
                            row[index],
                            STATE_LOG,
                            hls_path,
                        )
                    elif row[index].strip() == "-":
                        put(
                            result,
                            key,
                            None,
                            row[index],
                            STATE_MISSING,
                            hls_path,
                        )

    vivado_keys = [f"vivado.{resource}.used" for resource in VIVADO_RESOURCES]
    if any(
        result.metrics.get(key) is None
        or result.metrics[key].state == STATE_MISSING
        for key in vivado_keys
    ):
        vivado_path = resource_log_path(
            result.run_dir,
            "40_vivado_synth/utilization_post_synth.rpt",
            "utilization_post_synth.rpt",
        )
        if vivado_path:
            row = first_log_row(
                vivado_path,
                lambda cells: len(cells) > 1
                and normalize(cells[0]) == "snn_to_hls"
                and normalize(cells[1]) == "(top)",
            )
            if row is None or len(row) <= 10:
                result.warnings.append(
                    "Não foi possível localizar a linha top de recursos Vivado "
                    f"em {vivado_path}."
                )
            else:
                total_luts = log_integer(row[2])
                ffs = log_integer(row[6])
                ramb36 = log_integer(row[7])
                ramb18 = log_integer(row[8])
                uram = log_integer(row[9])
                dsp = log_integer(row[10])
                values: dict[str, int | Decimal | None] = {
                    "vivado.LUT.used": total_luts,
                    "vivado.FF.used": ffs,
                    "vivado.BRAM.used": (
                        Decimal(ramb36) + Decimal(ramb18) / Decimal(2)
                        if ramb36 is not None and ramb18 is not None
                        else None
                    ),
                    "vivado.DSP.used": dsp,
                    "vivado.URAM.used": uram,
                }
                raw_values = {
                    "vivado.LUT.used": row[2],
                    "vivado.FF.used": row[6],
                    "vivado.BRAM.used": f"RAMB36={row[7]}; RAMB18={row[8]}",
                    "vivado.DSP.used": row[10],
                    "vivado.URAM.used": row[9],
                }
                for key, value in values.items():
                    if value is None or result.metrics.get(key) is not None:
                        continue
                    put(
                        result,
                        key,
                        value,
                        raw_values[key],
                        STATE_LOG,
                        vivado_path,
                    )


def network_sort_key(network: str) -> tuple[int, int, str]:
    reset_group = 0 if "_zero" in network else 1 if "_subtract" in network else 2
    if network.startswith("time_driven"):
        method = 0
    elif network.endswith("_pwl"):
        method = 1
    elif network.endswith("_ts_efa"):
        method = 2
    else:
        method = 3
    return reset_group, method, network


def discover_latest_runs(
    runs_dir: Path,
    networks: set[str] | None = None,
    to_stage_override: str | None = None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for network_dir in sorted(runs_dir.iterdir()):
        network_match = NETWORK_RE.fullmatch(network_dir.name)
        if not network_dir.is_dir() or not network_match:
            continue
        if networks is not None and network_dir.name not in networks:
            continue
        candidates = [
            child
            for child in network_dir.iterdir()
            if child.is_dir() and RUN_RE.fullmatch(child.name)
        ]
        if not candidates:
            continue
        latest = max(
            candidates,
            key=lambda path: (RUN_RE.fullmatch(path.name).group(1), path.name),
        )
        summary_path = latest / "reports" / "summary.md"
        selection = RunResult(
            network=network_dir.name,
            backend=f"{network_match.group(1)}-driven",
            run_id=latest.name,
            run_dir=latest,
            summary_path=summary_path,
            summary_exists=summary_path.is_file(),
        )
        result = parse_summary(selection)
        if to_stage_override is not None:
            result.scope_to_stage = to_stage_override
        supplement_csim_accuracy_from_log(result)
        supplement_resources_from_logs(result)
        results.append(result)
    return sorted(results, key=lambda result: network_sort_key(result.network))


def format_pt(value: int | Decimal, decimals: int | None = None) -> str:
    decimal = Decimal(value)
    if decimals is None:
        decimals = max(0, -decimal.as_tuple().exponent)
    text = f"{decimal:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def format_metric(metric: Metric | None, spec: MetricSpec, *, suffix: bool = True) -> str:
    if metric is None or metric.state == STATE_MISSING:
        return "N/D"
    if metric.state == STATE_UNDEFINED:
        return "`undef`"
    if metric.value is None:
        return "N/D"
    if spec.kind == "text":
        rendered = str(metric.value)
        if spec.key in {"saif.file", "saif.strip_path"}:
            rendered = f"`{rendered}`"
    else:
        rendered = format_pt(metric.value, spec.decimals)
        if spec.unit == "%":
            rendered += "%"
    if metric.state == STATE_LOG:
        rendered += " §"
    if suffix and metric.state == STATE_PROVISIONAL:
        rendered += " ‡"
    return rendered


def short_label(result: RunResult) -> str:
    if result.network.startswith("time_driven_"):
        label = "TD " + result.network.removeprefix("time_driven_").replace("_", " ")
    else:
        variant = result.network.removeprefix("event_driven_")
        variant = variant.replace("_pwl", " PWL").replace("_ts_efa", " TS-EFA")
        label = "ED " + variant.replace("_", " ")
    if result.incomplete:
        label += " †"
    return label


def network_color(result: RunResult) -> str:
    if result.network.startswith("time_driven"):
        return "#4E79A7"
    if result.network.endswith("_pwl"):
        return "#59A14F"
    if result.network.endswith("_ts_efa"):
        return "#F28E2B"
    return "#B07AA1"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def metric_matrix(results: Sequence[RunResult], specs: Sequence[MetricSpec]) -> str:
    headers = ["Métrica", *(f"`{result.network}`" for result in results)]
    rows = []
    for spec in specs:
        label = spec.label + (f" ({spec.unit})" if spec.unit else "")
        rows.append(
            [
                label,
                *[
                    format_metric(result.metrics.get(spec.key), spec)
                    for result in results
                ],
            ]
        )
    return markdown_table(headers, rows)


def relative_link(target: Path, report_path: Path) -> str:
    return Path(os.path.relpath(target, report_path.parent)).as_posix()


def stage_summary(result: RunResult) -> str:
    if not result.summary_exists:
        return "summary ausente"
    failures = [
        f"`{stage}: {status}`"
        for stage in result.required_stages
        for status in [result.stages.get(stage, "N/D")]
        if normalize(status) not in {"success", "skipped"}
    ]
    if failures:
        return "; ".join(failures)
    endpoint = result.required_stages[-1]
    return "escopo até `{}` concluído".format(endpoint)


def report_stage_order(results: Sequence[RunResult]) -> list[str]:
    if not results:
        return STAGE_ORDER
    endpoint = max(
        STAGE_ORDER.index(result.required_stages[-1])
        for result in results
    )
    return STAGE_ORDER[: endpoint + 1]


def includes_post_synth_metrics(results: Sequence[RunResult]) -> bool:
    return any(result.includes_stage("post-synth-sim") for result in results)


def report_metric_specs(results: Sequence[RunResult]) -> list[MetricSpec]:
    if includes_post_synth_metrics(results):
        return ALL_SPECS
    excluded_groups = {
        "SAIF",
        "Duração e latência",
        "Potência",
        "Energia",
        "Energia calculada",
        "Qualidade SAIF",
    }
    return [spec for spec in ALL_SPECS if spec.group not in excluded_groups]


def display_project(result: RunResult, repo_root: Path) -> str:
    if not result.project:
        return "N/D"
    project_path = Path(result.project)
    try:
        return project_path.relative_to(repo_root).as_posix()
    except ValueError:
        return result.project


def log_source_rows(
    results: Sequence[RunResult],
    specs: Sequence[MetricSpec],
    report_path: Path,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for result in results:
        paths: list[Path] = []
        for spec in specs:
            metric = result.metrics.get(spec.key)
            if (
                metric
                and metric.state == STATE_LOG
                and metric.source
                and metric.source not in paths
            ):
                paths.append(metric.source)
        if paths:
            rows.append(
                [
                    f"`{result.network}`",
                    ", ".join(
                        f"[{path.name}]({relative_link(path, report_path)})"
                        for path in paths
                    ),
                ]
            )
    return rows


def same_workload(results: Sequence[RunResult]) -> bool:
    for spec in WORKLOAD_SPECS:
        values = [
            result.metrics.get(spec.key).value
            for result in results
            if result.metrics.get(spec.key)
        ]
        if len(values) != len(results) or len(set(values)) != 1:
            return False
    return bool(results)


def numeric_metric(result: RunResult, key: str) -> Decimal | None:
    metric = result.metrics.get(key)
    if not metric or not isinstance(metric.value, (int, Decimal)):
        return None
    return Decimal(metric.value)


def paired_comparison_rows(results: Sequence[RunResult]) -> list[list[str]]:
    by_name = {result.network: result for result in results}
    specs = [
        ("timing.latency_step_us", "Latência", "µs/step", 6),
        ("power.dynamic_w", "Potência dinâmica", "W", 3),
        ("energy.step_uj", "Energia", "µJ/step", 6),
    ]
    rows: list[list[str]] = []
    for reset in ["zero", "subtract"]:
        time_result = by_name.get(f"time_driven_{reset}")
        if not time_result:
            continue
        event_results = [
            result
            for result in results
            if result.network.startswith(f"event_driven_{reset}_")
        ]
        for event_result in event_results:
            for key, label, unit, decimals in specs:
                time_value = numeric_metric(time_result, key)
                event_value = numeric_metric(event_result, key)
                if time_value is None or event_value in {None, Decimal(0)}:
                    continue
                difference = (time_value / event_value - Decimal(1)) * Decimal(100)
                time_marker = (
                    " ‡"
                    if time_result.metrics[key].state == STATE_PROVISIONAL
                    else ""
                )
                event_marker = (
                    " ‡"
                    if event_result.metrics[key].state == STATE_PROVISIONAL
                    else ""
                )
                difference_marker = " ‡" if time_marker or event_marker else ""
                rows.append(
                    [
                        reset,
                        f"`{time_result.network}` vs. `{event_result.network}`",
                        f"{label} ({unit})",
                        format_pt(time_value, decimals) + time_marker,
                        format_pt(event_value, decimals) + event_marker,
                        format_pt(difference, 2) + "%" + difference_marker,
                    ]
                )
    return rows


def paired_csim_vivado_rows(results: Sequence[RunResult]) -> list[list[str]]:
    """Compara acurácia e recursos OOC entre redes do mesmo reset.

    A diferença de acurácia é aditiva (event-driven menos time-driven),
    portanto é apresentada em pontos percentuais. Para recursos, o overhead
    é relativo ao time-driven: ``(ED / TD - 1) * 100``. Comparações sem os
    dois operandos, ou cujo denominador de recurso seja zero, são omitidas.
    """

    by_name = {result.network: result for result in results}
    resource_specs = [
        ("vivado.LUT.used", "Vivado OOC — LUT"),
        ("vivado.FF.used", "Vivado OOC — FF"),
        ("vivado.BRAM.used", "Vivado OOC — BRAM"),
        ("vivado.DSP.used", "Vivado OOC — DSP"),
    ]
    rows: list[list[str]] = []
    for reset in ["zero", "subtract"]:
        time_result = by_name.get(f"time_driven_{reset}")
        if not time_result:
            continue
        event_results = [
            result
            for result in results
            if result.network.startswith(f"event_driven_{reset}_")
        ]
        for event_result in event_results:
            pair = f"`{time_result.network}` vs. `{event_result.network}`"
            time_accuracy = numeric_metric(time_result, "csim.accuracy_pct")
            event_accuracy = numeric_metric(event_result, "csim.accuracy_pct")
            if time_accuracy is not None and event_accuracy is not None:
                difference_pp = event_accuracy - time_accuracy
                rows.append(
                    [
                        reset,
                        pair,
                        "Acurácia CSim",
                        format_metric(
                            time_result.metrics["csim.accuracy_pct"],
                            CSIM_SPECS[0],
                        ),
                        format_metric(
                            event_result.metrics["csim.accuracy_pct"],
                            CSIM_SPECS[0],
                        ),
                        format_pt(difference_pp, 2) + " p.p.",
                    ]
                )

            for key, label in resource_specs:
                time_value = numeric_metric(time_result, key)
                event_value = numeric_metric(event_result, key)
                if time_value in {None, Decimal(0)} or event_value is None:
                    continue
                overhead = (event_value / time_value - Decimal(1)) * Decimal(100)
                spec = next(spec for spec in VIVADO_USED_SPECS if spec.key == key)
                rows.append(
                    [
                        reset,
                        pair,
                        label,
                        format_metric(time_result.metrics[key], spec),
                        format_metric(event_result.metrics[key], spec),
                        format_pt(overhead, decimals=2) + "%",
                    ]
                )
    return rows


def append_report_footer(
    lines: list[str],
    results: Sequence[RunResult],
    include_post_synth: bool,
    report_path: Path,
    repo_root: Path,
) -> None:
    lines.extend(
        [
            "## Limitações",
            "",
            "1. O relatório usa apenas um run por variante e não mede variabilidade.",
            "2. Métrica ausente no summary só é preenchida quando o log do próprio",
            "   run selecionado fornece um valor explícito; nenhum run antigo é usado.",
            "3. Estimativas HLS e recursos Vivado OOC pertencem a fases diferentes",
            "   e permanecem separados.",
        ]
    )
    if include_post_synth:
        lines.extend(
            [
                "4. Transições SAIF devem ser interpretadas junto com duração, cobertura",
                "   e total de nets.",
            ]
        )
        partial_limits = sorted(
            {
                result.capture_transaction_limit
                for result in results
                if result.capture_is_partial and result.capture_transaction_limit
            }
        )
        if partial_limits:
            limits = ", ".join(str(limit) for limit in partial_limits)
            lines.extend(
                [
                    "5. A captura pós-síntese foi parcial e limitada a "
                    f"{limits} transação(ões) por run; potência, energia e",
                    "   latência baseada em SAIF são estimativas dessa janela e não",
                    "   representam o workload completo.",
                    "6. Potência e energia marcadas com ‡ são estimativas provisórias.",
                    "7. Recursos marcados com § vêm dos logs HLS/Vivado e não incluem",
                    "   capacidades que não foram reportadas nesses arquivos.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "5. Potência e energia marcadas com ‡ são estimativas provisórias.",
                    "6. Recursos marcados com § vêm dos logs HLS/Vivado e não incluem",
                    "   capacidades que não foram reportadas nesses arquivos.",
                    "",
                ]
            )
    else:
        final_scopes = sorted(
            {result.required_stages[-1] for result in results},
            key=STAGE_ORDER.index,
        )
        scope_text = ", ".join(f"`{stage}`" for stage in final_scopes)
        lines.extend(
            [
                f"4. Os escopos selecionados terminam em {scope_text}; nenhuma",
                "   métrica posterior ao escopo de cada run é presumida.",
                "5. Recursos marcados com § vêm dos logs HLS/Vivado do próprio run.",
                "6. Utilização OOC não implica fechamento de timing; o relatório",
                "   `timing_post_synth.rpt` deve ser analisado separadamente.",
                "",
            ]
        )
    warnings = [
        f"`{result.network}`: {warning}"
        for result in results
        for warning in result.warnings
    ]
    if warnings:
        lines.extend(
            [
                "## Avisos do gerador",
                "",
                *[f"- {warning}" for warning in warnings],
                "",
            ]
        )
    lines.extend(
        [
            "## Reprodução",
            "",
            "```bash",
            "MPLCONFIGDIR=/tmp/neurohls-results-mpl "
            + shlex.join(
                [
                    ".venv/bin/python",
                    "results/gerar_resultados.py",
                    "--output-dir",
                    os.path.relpath(report_path.parent, repo_root),
                    *(
                        ["--to-stage", results[0].required_stages[-1]]
                        if results
                        and len(
                            {
                                result.required_stages[-1]
                                for result in results
                                if result.required_stages
                            }
                        )
                        == 1
                        and results[0].required_stages[-1] != STAGE_ORDER[-1]
                        else []
                    ),
                    *[
                        argument
                        for result in results
                        for argument in ("--network", result.network)
                    ],
                ]
            ),
            "```",
            "",
        ]
    )


def render_report(
    results: Sequence[RunResult],
    report_path: Path,
    csv_path: Path,
    repo_root: Path,
) -> str:
    graph = lambda name: f"graficos/{name}"
    backends = {result.backend for result in results}
    if backends == {"event-driven"}:
        title = "# Resultados dos runs event-driven mais recentes"
    elif backends == {"time-driven"}:
        title = "# Resultados dos runs time-driven mais recentes"
    else:
        title = "# Comparação dos runs event-driven e time-driven mais recentes"
    lines: list[str] = [
        title,
        "",
        "Relatório gerado automaticamente por `results/gerar_resultados.py`.",
        "Para cada variante selecionada em `sim/runs`, foi escolhido exclusivamente",
        "o diretório com o timestamp UTC mais recente. O gerador não recua para",
        "um run antigo quando o summary do run mais novo está ausente.",
        "",
        "`N/D` significa que a métrica não aparece no `summary.md` selecionado;",
        "`undef` preserva uma indefinição explicitamente reportada. `†` marca um",
        "run com etapa requerida não concluída, `‡` marca potência/energia provisória e",
        "`§` marca um valor extraído diretamente de um log do run selecionado.",
        "",
        f"Os dados normalizados também estão em [{csv_path.name}]({csv_path.name}).",
        "",
        "## Runs selecionados",
        "",
    ]

    run_rows = []
    for result in results:
        source = (
            f"[summary.md]({relative_link(result.summary_path, report_path)})"
            if result.summary_exists
            else "N/D"
        )
        run_rows.append(
            [
                f"`{result.network}`" + (" †" if result.incomplete else ""),
                result.backend,
                f"`{result.run_id}`",
                f"`{display_project(result, repo_root)}`" if result.project else "N/D",
                "`{}`".format(result.required_stages[-1]),
                source,
                stage_summary(result),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Variante",
                    "Backend",
                    "Run mais recente",
                    "Projeto",
                    "Escopo até",
                    "Fonte",
                    "Estado",
                ],
                run_rows,
            ),
            "",
            "## Plataforma",
            "",
            markdown_table(
                ["Variante", "Top", "FPGA", "Clock", "Período"],
                [
                    [
                        f"`{result.network}`",
                        f"`{result.top}`" if result.top else "N/D",
                        f"`{result.fpga}`" if result.fpga else "N/D",
                        (
                            f"{format_pt(result.clock_mhz, 0)} MHz"
                            if result.clock_mhz is not None
                            else "N/D"
                        ),
                        (
                            f"{format_pt(result.clock_ns, 5)} ns"
                            if result.clock_ns is not None
                            else "N/D"
                        ),
                    ]
                    for result in results
                ],
            ),
            "",
            "## Estado das etapas",
            "",
            markdown_table(
                ["Etapa", *(f"`{result.network}`" for result in results)],
                [
                    [
                        f"`{stage}`",
                        *[
                            result.stages.get(stage, "N/D")
                            for result in results
                        ],
                    ]
                    for stage in report_stage_order(results)
                ],
            ),
            "",
            f"![Estado das etapas]({graph('01_etapas.svg')})",
            "",
            "## Acurácia CSim",
            "",
            metric_matrix(results, CSIM_SPECS),
            "",
            "A acurácia é extraída da linha `Final Acc` do `csim.log` pertencente",
            "ao próprio run selecionado. Quando essa linha não existe, o valor",
            "permanece N/D; o gerador não consulta runs anteriores.",
            "",
            "## Carga da simulação",
            "",
            metric_matrix(results, WORKLOAD_SPECS),
            "",
        ]
    )
    if same_workload(results):
        first = results[0]
        samples = format_metric(
            first.metrics.get("workload.samples"),
            WORKLOAD_SPECS[0],
        )
        steps = format_metric(
            first.metrics.get("workload.steps_per_sample"),
            WORKLOAD_SPECS[1],
        )
        total = format_metric(
            first.metrics.get("workload.steps_total"),
            WORKLOAD_SPECS[2],
        )
        lines.extend(
            [
                f"Todos os runs usam a mesma carga lógica: {samples} amostras ×",
                f"{steps} passos = {total} passos executados.",
                "",
            ]
        )
    lines.extend(
        [
            "O batch organiza o testbench e não representa paralelismo do hardware.",
            "Em backends event-driven, o número de transações do DUT pode ser",
            "diferente do número de passos temporais lógicos.",
            "",
            f"![Carga da simulação]({graph('04_carga_simulacao.svg')})",
            "",
            "## Estimativa de recursos HLS",
            "",
            "Estes valores são anteriores à síntese Vivado.",
            "",
            metric_matrix(results, HLS_SPECS),
            "",
            metric_matrix(results, HLS_LATENCY_SPECS),
            "",
        ]
    )
    missing_hls = [
        result.network
        for result in results
        if all(result.metrics.get(spec.key) is None for spec in HLS_SPECS)
    ]
    if missing_hls:
        lines.extend(
            [
                "Os summaries mais recentes de "
                + ", ".join(f"`{name}`" for name in missing_hls)
                + " não contêm a seção de recursos HLS; os valores permanecem N/D.",
                "",
            ]
        )
    hls_log_rows = log_source_rows(results, HLS_SPECS, report_path)
    if hls_log_rows:
        lines.extend(
            [
                "Os valores de recursos HLS marcados com `§` foram extraídos do",
                "`csynth.rpt` do run mais recente. Quando o relatório mostra `-`,",
                "o valor é mantido como N/D, sem convertê-lo artificialmente em zero.",
                "",
                markdown_table(["Variante", "Fonte HLS"], hls_log_rows),
                "",
            ]
        )
    lines.extend(
        [
            f"![Estimativa de recursos HLS]({graph('02_recursos_hls.svg')})",
            "",
            "## Uso de recursos pós-síntese — Vivado OOC",
            "",
            "### Quantidade utilizada",
            "",
            metric_matrix(results, VIVADO_USED_SPECS),
            "",
        ]
    )
    vivado_log_rows = log_source_rows(results, VIVADO_USED_SPECS, report_path)
    if vivado_log_rows:
        lines.extend(
            [
                "Os valores utilizados marcados com `§` foram extraídos do",
                "relatório `utilization_post_synth.rpt` do run mais recente.",
                "BRAM é normalizado como `RAMB36 + RAMB18 / 2`; no log dos",
                "time-driven isso corresponde a `7 + 47 / 2 = 30,5` blocos.",
                "",
                markdown_table(["Variante", "Fonte Vivado"], vivado_log_rows),
                "",
            ]
        )
    lines.extend(
        [
            f"![Recursos Vivado OOC]({graph('03_recursos_vivado_ooc.svg')})",
            "",
            "### Capacidade e percentual reportados",
            "",
        ]
    )
    vivado_capacity_rows = []
    for resource in VIVADO_RESOURCES:
        row = [resource]
        available_spec = next(
            spec
            for spec in ALL_SPECS
            if spec.key == f"vivado.{resource}.available"
        )
        usage_spec = next(
            spec
            for spec in ALL_SPECS
            if spec.key == f"vivado.{resource}.usage_pct"
        )
        for result in results:
            available = format_metric(
                result.metrics.get(available_spec.key),
                available_spec,
            )
            usage = format_metric(
                result.metrics.get(usage_spec.key),
                usage_spec,
            )
            row.append(f"{available} / {usage}")
        vivado_capacity_rows.append(row)
    lines.extend(
        [
            "Cada célula mostra `Disponível / Uso`.",
            "",
            markdown_table(
                ["Recurso", *(f"`{result.network}`" for result in results)],
                vivado_capacity_rows,
            ),
            "",
            "Capacidade igual a zero não fornece um denominador válido para",
            "comparar ocupação. Esses percentuais são preservados, mas não entram",
            "nos gráficos nem nos rankings; os logs suplementares fornecem apenas",
            "as quantidades utilizadas, sem capacidade disponível.",
            "",
        ]
    )
    post_synth_in_scope = includes_post_synth_metrics(results)
    if not post_synth_in_scope:
        paired_rows = paired_csim_vivado_rows(results)
        if paired_rows:
            lines.extend(
                [
                    "## Comparação pareada time-driven × event-driven",
                    "",
                    "Para acurácia, a diferença é `ED - TD` e usa pontos",
                    "percentuais. Para recursos, o overhead é calculado como",
                    "`(ED / TD - 1) × 100`; valor positivo indica maior uso de",
                    "recursos pelo event-driven. Comparações com valor ausente ou",
                    "denominador zero são omitidas.",
                    "",
                    markdown_table(
                        [
                            "Reset",
                            "Par",
                            "Métrica",
                            "Time-driven",
                            "Event-driven",
                            "Diferença / overhead",
                        ],
                        paired_rows,
                    ),
                    "",
                ]
            )
        final_scopes = sorted(
            {result.required_stages[-1] for result in results},
            key=STAGE_ORDER.index,
        )
        if len(final_scopes) == 1:
            scope_description = (
                "Estes runs foram avaliados somente até "
                f"`{final_scopes[0]}`."
            )
        else:
            scope_description = (
                "Os runs têm escopos finais distintos: "
                + ", ".join(f"`{stage}`" for stage in final_scopes)
                + ". A tabela de runs identifica o limite de cada variante."
            )
        lines.extend(
            [
                "## Encerramento intencional do fluxo",
                "",
                scope_description,
                "Nenhum run selecionado inclui `post-synth-sim` ou `power`; por isso",
                "o relatório não apresenta tabelas ou gráficos de SAIF, latência",
                "derivada da atividade, potência ou energia.",
                "",
            ]
        )
        append_report_footer(
            lines,
            results,
            include_post_synth=False,
            report_path=report_path,
            repo_root=repo_root,
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Atividade SAIF",
            "",
            metric_matrix(results, SAIF_SPECS),
            "",
            markdown_table(
                ["Variante", "Arquivo SAIF", "Strip path"],
                [
                    [
                        f"`{result.network}`",
                        (
                            f"[`post_synth_activity.saif`]"
                            f"({relative_link(Path(result.metrics['saif.file'].value), report_path)})"
                            if result.metrics.get("saif.file")
                            and Path(str(result.metrics["saif.file"].value)).exists()
                            else format_metric(
                                result.metrics.get("saif.file"),
                                EXTRA_SPECS[0],
                            )
                        ),
                        format_metric(
                            result.metrics.get("saif.strip_path"),
                            EXTRA_SPECS[1],
                        ),
                    ]
                    for result in results
                ],
            ),
            "",
            "O campo de duração SAIF não informa unidade junto ao valor bruto.",
            "Os cálculos dos próprios summaries o tratam como picosegundos.",
            "",
        ]
    )
    partial_limits = sorted(
        {
            result.capture_transaction_limit
            for result in results
            if result.capture_is_partial and result.capture_transaction_limit
        }
    )
    if partial_limits:
        limits = ", ".join(str(limit) for limit in partial_limits)
        lines.extend(
            [
                "A captura pós-síntese foi parcial: cada run foi limitado a "
                f"{limits} transação(ões) por `SIM_POST_SYNTH_TRANSACTION_LIMIT`.",
                "Os valores de atividade, latência, potência e energia devem ser",
                "interpretados como estimativas da janela capturada; a carga lógica",
                "do testbench continua registrada como 1.792 passos (7 × 256).",
                "",
            ]
        )
    lines.extend(
        [
            f"![Atividade SAIF]({graph('05_atividade_saif.svg')})",
            "",
            "## Duração e latência",
            "",
            metric_matrix(results, TIMING_SPECS),
            "",
            "As latências são médias amortizadas sobre toda a janela SAIF. Elas",
            "incluem inicialização, intervalos e finalização; não são uma medição",
            "isolada de `ap_start` até `ap_done` nem wall-clock do simulador.",
            "",
            f"![Duração e latência]({graph('06_duracao_latencia.svg')})",
            "",
            "## Potência",
            "",
            metric_matrix(results, POWER_SPECS),
            "",
        ]
    )
    provisional = [result.network for result in results if result.power_provisional]
    if provisional:
        lines.extend(
            [
                "Os valores marcados com ‡ são mantidos porque aparecem no summary,",
                "mas são provisórios. Runs afetados: "
                + ", ".join(f"`{name}`" for name in provisional)
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            f"![Potência]({graph('07_potencia.svg')})",
            "",
            "## Energia",
            "",
            "### Valores apresentados nas tabelas dos summaries",
            "",
            metric_matrix(results, ENERGY_SPECS),
            "",
            f"![Energia]({graph('08_energia.svg')})",
            "",
            "### Valores preservados nos blocos de cálculo",
            "",
            metric_matrix(results, ENERGY_CALC_SPECS),
            "",
            "Os summaries calculam `E_total = P_total × duração`, depois dividem",
            "essa energia pelos passos executados e pelas amostras.",
            "",
            "## Qualidade da estimativa de potência",
            "",
            metric_matrix(results, QUALITY_SPECS),
            "",
            "A cobertura é uma métrica de qualidade da anotação, não de desempenho.",
            "Cada run mantém seu próprio limite; por isso o gráfico compara duas",
            "barras por rede em vez de usar uma linha fixa de 50%.",
            "",
            f"![Qualidade SAIF]({graph('09_qualidade_saif.svg')})",
            "",
            "## Comparação pareada time-driven × event-driven",
            "",
        ]
    )
    paired_rows = paired_comparison_rows(results)
    if paired_rows:
        lines.extend(
            [
                "A diferença é calculada como `(TD / ED - 1) × 100`; valor negativo",
                "indica que o time-driven reportou um valor menor.",
                "",
                markdown_table(
                    [
                        "Reset",
                        "Par",
                        "Métrica",
                        "Time-driven",
                        "Event-driven",
                        "Diferença",
                    ],
                    paired_rows,
                ),
                "",
            ]
        )
    else:
        lines.extend(["Não há pares suficientes nos summaries selecionados.", ""])

    append_report_footer(
        lines,
        results,
        include_post_synth=True,
        report_path=report_path,
        repo_root=repo_root,
    )
    return "\n".join(lines)


def normalized_value(metric: Metric | None) -> str:
    if metric is None or metric.value is None:
        return ""
    if isinstance(metric.value, Decimal):
        return format(metric.value, "f")
    return str(metric.value)


def render_csv(results: Sequence[RunResult], repo_root: Path) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "rede",
        "backend",
        "run",
        "grupo",
        "metrica",
        "valor_normalizado",
        "valor_original",
        "unidade",
        "estado",
        "fonte",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    for result in results:
        source = ""
        if result.summary_exists:
            try:
                source = result.summary_path.relative_to(repo_root).as_posix()
            except ValueError:
                source = result.summary_path.as_posix()

        def write_row(
            group: str,
            metric_name: str,
            value: str,
            raw: str,
            unit: str,
            state: str,
            metric: Metric | None = None,
        ) -> None:
            metric_source = metric.source if metric and metric.source else None
            source_path = metric_source or (
                result.summary_path if result.summary_exists else None
            )
            row_source = ""
            if source_path:
                try:
                    row_source = source_path.relative_to(repo_root).as_posix()
                except ValueError:
                    row_source = source_path.as_posix()
            writer.writerow(
                {
                    "rede": result.network,
                    "backend": result.backend,
                    "run": result.run_id,
                    "grupo": group,
                    "metrica": metric_name,
                    "valor_normalizado": value,
                    "valor_original": raw,
                    "unidade": unit,
                    "estado": state,
                    "fonte": row_source,
                }
            )

        metadata_rows = [
            ("Projeto", result.project),
            ("Top", result.top),
            ("FPGA", result.fpga),
            ("Escopo até", result.required_stages[-1]),
            (
                "Clock",
                format(result.clock_mhz, "f") if result.clock_mhz is not None else None,
            ),
            (
                "Período do clock",
                format(result.clock_ns, "f") if result.clock_ns is not None else None,
            ),
        ]
        for name, value in metadata_rows:
            unit = "MHz" if name == "Clock" else "ns" if name == "Período do clock" else ""
            write_row(
                "Metadados",
                name,
                value or "",
                value or "",
                unit,
                STATE_REPORTED if value is not None else STATE_MISSING,
            )
        for stage in report_stage_order(results):
            status = result.stages.get(stage)
            write_row(
                "Etapas",
                stage,
                status or "",
                status or "",
                "",
                STATE_REPORTED if status is not None else STATE_MISSING,
            )
        for spec in report_metric_specs(results):
            metric = result.metrics.get(spec.key)
            write_row(
                spec.group,
                spec.label,
                normalized_value(metric),
                metric.raw if metric and metric.raw else "",
                spec.unit,
                metric.state if metric else STATE_MISSING,
                metric,
            )
    return buffer.getvalue()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def chart_footer(
    fig: plt.Figure,
    results: Sequence[RunResult],
    specs: Sequence[MetricSpec],
) -> None:
    parts: list[str] = []
    if any(result.incomplete for result in results):
        parts.append("† run com etapa não concluída")
    if any(
        result.metrics.get(spec.key)
        and result.metrics[spec.key].state == STATE_PROVISIONAL
        for result in results
        for spec in specs
    ):
        parts.append("‡ valor provisório")
    if any(
        result.metrics.get(spec.key)
        and result.metrics[spec.key].state == STATE_LOG
        for result in results
        for spec in specs
    ):
        parts.append("§ valor extraído do log")
    if any(
        result.metrics.get(spec.key) is None
        or result.metrics[spec.key].state in {STATE_MISSING, STATE_UNDEFINED}
        for result in results
        for spec in specs
    ):
        parts.append("N/D = ausente; undef = indefinido no summary")
    if not parts:
        return
    fig.text(
        0.5,
        0.012,
        "  •  ".join(parts),
        ha="center",
        va="bottom",
        color="#666666",
        fontsize=8.5,
    )


def metric_number(result: RunResult, spec: MetricSpec) -> float | None:
    metric = result.metrics.get(spec.key)
    if not metric or not isinstance(metric.value, (int, Decimal)):
        return None
    return float(metric.value)


def graph_label(metric: Metric, spec: MetricSpec) -> str:
    rendered = format_metric(metric, spec, suffix=False)
    if metric.state == STATE_PROVISIONAL:
        rendered += " ‡"
    return rendered


def bar_panel(
    ax: plt.Axes,
    results: Sequence[RunResult],
    spec: MetricSpec,
    *,
    xlim: tuple[float, float] | None = None,
) -> None:
    values = [metric_number(result, spec) for result in results]
    valid = [value for value in values if value is not None]
    maximum = max(valid, default=0.0)
    scale = maximum if maximum > 0 else 1.0
    plotted = [0.0 if value is None else value for value in values]
    bars = ax.barh(
        range(len(results)),
        plotted,
        color=[network_color(result) for result in results],
        height=0.62,
    )

    if xlim is None:
        ax.set_xlim(0, scale * 1.29)
    else:
        ax.set_xlim(*xlim)
        scale = max(scale, xlim[1] - xlim[0])
    ax.set_yticks(
        range(len(results)),
        labels=[short_label(result) for result in results],
    )
    ax.invert_yaxis()
    title = spec.label + (f" ({spec.unit})" if spec.unit else "")
    ax.set_title(title, loc="left", pad=9)
    style_axis(ax)

    offset = scale * 0.018
    for bar, result, value in zip(bars, results, values):
        metric = result.metrics.get(spec.key)
        y = bar.get_y() + bar.get_height() / 2
        if value is None or metric is None:
            text = "undef" if metric and metric.state == STATE_UNDEFINED else "N/D"
            ax.text(
                offset,
                y,
                text,
                va="center",
                ha="left",
                color="#777777",
                fontstyle="italic",
                fontweight="bold",
            )
            continue
        ax.text(
            value + offset,
            y,
            graph_label(metric, spec),
            va="center",
            ha="left",
            color="#222222",
            fontsize=9,
        )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(
        temporary,
        format="svg",
        bbox_inches="tight",
        metadata={
            "Date": None,
            "Creator": "NeuroHLS results/gerar_resultados.py",
        },
    )
    os.replace(temporary, path)
    plt.close(fig)


def plot_grid(
    results: Sequence[RunResult],
    specs: Sequence[MetricSpec],
    path: Path,
    title: str,
    columns: int,
) -> None:
    rows = math.ceil(len(specs) / columns)
    figure_height = 5.3 if rows == 1 else 3.6 * rows + 0.8
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.1 * columns, figure_height),
        squeeze=False,
    )
    fig.suptitle(title, x=0.055, ha="left", fontsize=15, fontweight="bold")
    for ax, spec in zip(axes.flat, specs):
        bar_panel(ax, results, spec)
    for ax in axes.flat[len(specs) :]:
        ax.set_visible(False)
    chart_footer(fig, results, specs)
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        top=0.80 if rows == 1 else 0.90,
        bottom=0.14 if rows == 1 else 0.075,
        hspace=0.48,
        wspace=0.34,
    )
    save_figure(fig, path)


def plot_stages(results: Sequence[RunResult], path: Path) -> None:
    stages = report_stage_order(results)
    status_code = {
        "missing": 0,
        "not-run": 1,
        "failed": 2,
        "success": 3,
    }
    status_text = {
        "missing": "N/D",
        "not-run": "não exec.",
        "failed": "falhou",
        "success": "OK",
    }

    def category(status: str | None) -> str:
        if status is None:
            return "missing"
        normalized = normalize(status)
        if normalized == "success":
            return "success"
        if normalized == "failed":
            return "failed"
        return "not-run"

    matrix = [
        [
            status_code[category(result.stages.get(stage))]
            for stage in stages
        ]
        for result in results
    ]
    fig, ax = plt.subplots(figsize=(13.6, 5.2))
    cmap = ListedColormap(["#ECECEC", "#C9C9C9", "#E15759", "#59A14F"])
    ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(
        range(len(stages)),
        labels=stages,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        range(len(results)),
        labels=[short_label(result) for result in results],
    )
    ax.set_title("Estado das etapas no run mais recente", loc="left", pad=14)
    ax.tick_params(length=0)
    for row, result in enumerate(results):
        for column, stage in enumerate(stages):
            status = category(result.stages.get(stage))
            ax.text(
                column,
                row,
                status_text[status],
                ha="center",
                va="center",
                color="white" if status in {"success", "failed"} else "#555555",
                fontsize=8,
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    chart_footer(fig, results, [])
    fig.subplots_adjust(bottom=0.24, top=0.90, left=0.16, right=0.98)
    save_figure(fig, path)


def plot_quality(results: Sequence[RunResult], path: Path) -> None:
    annotated_spec = QUALITY_SPECS[1]
    total_spec = QUALITY_SPECS[2]
    coverage_spec = QUALITY_SPECS[3]
    threshold_spec = QUALITY_SPECS[4]
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4))
    fig.suptitle(
        "Qualidade da anotação SAIF",
        x=0.055,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    bar_panel(axes[0], results, annotated_spec)
    bar_panel(axes[1], results, total_spec)

    y = list(range(len(results)))
    coverage = [metric_number(result, coverage_spec) for result in results]
    threshold = [metric_number(result, threshold_spec) for result in results]
    height = 0.34
    axes[2].barh(
        [position - height / 2 for position in y],
        [0 if value is None else value for value in coverage],
        height=height,
        color="#4E79A7",
        label="cobertura",
    )
    axes[2].barh(
        [position + height / 2 for position in y],
        [0 if value is None else value for value in threshold],
        height=height,
        color="#BAB0AC",
        label="limite",
    )
    axes[2].set_yticks(y, labels=[short_label(result) for result in results])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 108)
    axes[2].set_title("Cobertura e limite (%)", loc="left", pad=9)
    style_axis(axes[2])
    axes[2].legend(frameon=False, loc="lower right", fontsize=8.5)
    for position, (result, coverage_value, threshold_value) in enumerate(
        zip(results, coverage, threshold)
    ):
        if coverage_value is None:
            axes[2].text(
                1.5,
                position,
                "N/D",
                va="center",
                color="#777777",
                fontstyle="italic",
                fontweight="bold",
            )
            continue
        axes[2].text(
            coverage_value + 1.2,
            position - height / 2,
            format_metric(
                result.metrics.get(coverage_spec.key),
                coverage_spec,
                suffix=False,
            ).removesuffix("%"),
            va="center",
            fontsize=8.5,
        )
        if threshold_value is not None:
            axes[2].text(
                threshold_value + 1.2,
                position + height / 2,
                format_metric(
                    result.metrics.get(threshold_spec.key),
                    threshold_spec,
                    suffix=False,
                ).removesuffix("%"),
                va="center",
                fontsize=8.5,
            )
    chart_footer(fig, results, QUALITY_SPECS)
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.82,
        bottom=0.13,
        wspace=0.42,
    )
    save_figure(fig, path)


def generate_charts(results: Sequence[RunResult], graphs_dir: Path) -> None:
    plot_stages(results, graphs_dir / "01_etapas.svg")
    plot_grid(
        results,
        HLS_SPECS,
        graphs_dir / "02_recursos_hls.svg",
        "Estimativa de recursos HLS (pré-síntese Vivado)",
        2,
    )
    plot_grid(
        results,
        VIVADO_USED_SPECS,
        graphs_dir / "03_recursos_vivado_ooc.svg",
        "Recursos utilizados após síntese Vivado OOC",
        2,
    )
    plot_grid(
        results,
        WORKLOAD_SPECS,
        graphs_dir / "04_carga_simulacao.svg",
        "Carga lógica da simulação",
        2,
    )
    if not includes_post_synth_metrics(results):
        return
    plot_grid(
        results,
        [
            MetricSpec(
                "timing.duration_ms",
                "SAIF",
                "Duração lógica total",
                "ms",
                9,
            ),
            SAIF_SPECS[1],
        ],
        graphs_dir / "05_atividade_saif.svg",
        "Atividade SAIF",
        2,
    )
    plot_grid(
        results,
        [TIMING_SPECS[0], TIMING_SPECS[2], TIMING_SPECS[3]],
        graphs_dir / "06_duracao_latencia.svg",
        "Duração e latência média amortizada",
        3,
    )
    plot_grid(
        results,
        POWER_SPECS,
        graphs_dir / "07_potencia.svg",
        "Potência reportada pelo Vivado",
        3,
    )
    plot_grid(
        results,
        ENERGY_SPECS,
        graphs_dir / "08_energia.svg",
        "Energia calculada no summary",
        3,
    )
    plot_quality(results, graphs_dir / "09_qualidade_saif.svg")


def validate_results(results: Sequence[RunResult], strict: bool) -> None:
    if not results:
        raise RuntimeError("Nenhuma variante event_driven/time_driven encontrada.")
    if strict:
        problems = [
            f"{result.network}: {warning}"
            for result in results
            for warning in result.warnings
        ]
        if problems:
            raise RuntimeError("Avisos em modo estrito:\n" + "\n".join(problems))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "sim" / "runs",
        help="Diretório contendo as variantes e os runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Diretório de saída do relatório, CSV e gráficos.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falha se houver summary ausente ou aviso de parsing.",
    )
    parser.add_argument(
        "--network",
        action="append",
        dest="networks",
        help=(
            "Inclui somente esta variante; pode ser repetido. A seleção "
            "continua usando exclusivamente o run mais recente de cada variante."
        ),
    )
    parser.add_argument(
        "--to-stage",
        choices=STAGE_ORDER,
        help=(
            "Define explicitamente a etapa final intencional dos runs selecionados. "
            "Útil para runs legados cujo summary ainda não registra o escopo."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = output_dir / "metricas_compiladas.csv"
    graphs_dir = output_dir / "graficos"

    network_filter = set(args.networks) if args.networks else None
    results = discover_latest_runs(
        runs_dir,
        network_filter,
        to_stage_override=args.to_stage,
    )
    validate_results(results, args.strict)
    backends = {result.backend for result in results}
    if backends == {"event-driven"}:
        report_name = "comparativo_event_driven.md"
    elif backends == {"time-driven"}:
        report_name = "comparativo_time_driven.md"
    else:
        report_name = "comparativo_event_time_driven.md"
    report_path = output_dir / report_name
    generate_charts(results, graphs_dir)
    atomic_write(csv_path, render_csv(results, repo_root))
    atomic_write(
        report_path,
        render_report(results, report_path, csv_path, repo_root),
    )

    print(f"Variantes compiladas: {len(results)}")
    for result in results:
        print(f"- {result.network}: {result.run_id}")
    print(f"Relatório: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Gráficos: {graphs_dir}")


if __name__ == "__main__":
    main()
