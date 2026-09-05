"""Orquestração do fluxo Vitis HLS -> Vivado -> SAIF -> potência.

O módulo suporta o layout clássico de uma pasta gerada pelo NeuroHLS. Ele não
gera redes a partir de NIR: a entrada é copiada de forma isolada e tratada como
somente leitura.
"""

import json
import math
import os
import re
import shlex
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import RunConfig
from .errors import SimEnvironmentError, StageError, ValidationError
from .project import (
    MEDOID_TESTBENCH_FILES,
    ProjectInfo,
    copy_project,
    validate_project,
)
from .tcl import (
    clock_override_xdc,
    cosim_tcl,
    cosim_setup_tcl,
    export_tcl,
    power_tcl,
    vitis_wrapper,
    vivado_synth_tcl,
    xsim_saif_tcl,
)
from .utils import (
    CommandRunner,
    atomic_write_json,
    atomic_write_text,
    command_text,
    read_json,
    run_stamp,
    sha256_file,
    slug,
    tcl_quote,
    utc_now,
)


STAGE_ORDER = (
    "prepare",
    "vitis-project",
    "csim",
    "hls-synth",
    "cosim-setup",
    "cosim",
    "export",
    "vivado-synth",
    "post-synth-sim",
    "power",
)


_SAIF_TIME_UNIT_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "fs": 1e-15,
}


def saif_duration_seconds(duration: Any, timescale: Optional[str]) -> Optional[float]:
    """Converte ``DURATION`` do SAIF para segundos sem assumir uma unidade fixa."""
    if timescale is None:
        return None
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(s|ms|us|µs|ns|ps|fs)\s*",
        str(timescale),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        numeric_duration = float(duration)
        multiplier = float(match.group(1))
    except (TypeError, ValueError):
        return None
    unit = match.group(2).lower()
    result = numeric_duration * multiplier * _SAIF_TIME_UNIT_SECONDS[unit]
    return result if math.isfinite(result) and result > 0 else None


@dataclass
class SaifInfo:
    path: Path
    size_bytes: int
    duration: float
    timescale: Optional[str]
    transition_count: int

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "duration": self.duration,
            "timescale": self.timescale,
            "transition_count": self.transition_count,
            "sha256": sha256_file(self.path),
        }
        duration_seconds = saif_duration_seconds(self.duration, self.timescale)
        if duration_seconds is not None:
            result["duration_seconds"] = duration_seconds
        return result


@dataclass
class WorkloadInfo:
    declared_samples: int
    executed_samples: int
    batch_size: int
    batch_count: int
    steps_per_sample: int
    total_logical_steps: int
    ignored_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "declared_samples": self.declared_samples,
            "executed_samples": self.executed_samples,
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "steps_per_sample": self.steps_per_sample,
            "total_logical_steps": self.total_logical_steps,
            "ignored_samples": self.ignored_samples,
            "source": "project/testbench.cpp",
        }


def parse_testbench_workload(path: Path) -> Optional[WorkloadInfo]:
    """Extrai a carga efetivamente percorrida pelo testbench clássico NeuroHLS."""
    path = Path(path)
    if not path.is_file():
        return None
    contents = path.read_text(encoding="utf-8", errors="replace")
    values: Dict[str, int] = {}
    for name in ("TOTAL_SAMPLES", "BATCH_SIZE", "STEP_COUNT"):
        match = re.search(
            r"^\s*#\s*define\s+{}\s+\(?\s*([0-9]+)\s*\)?(?:\s|$)".format(
                re.escape(name)
            ),
            contents,
            flags=re.MULTILINE,
        )
        if match is None:
            return None
        values[name] = int(match.group(1))
    if any(value <= 0 for value in values.values()):
        return None
    batch_count = values["TOTAL_SAMPLES"] // values["BATCH_SIZE"]
    executed_samples = batch_count * values["BATCH_SIZE"]
    if executed_samples <= 0:
        return None
    return WorkloadInfo(
        declared_samples=values["TOTAL_SAMPLES"],
        executed_samples=executed_samples,
        batch_size=values["BATCH_SIZE"],
        batch_count=batch_count,
        steps_per_sample=values["STEP_COUNT"],
        total_logical_steps=executed_samples * values["STEP_COUNT"],
        ignored_samples=values["TOTAL_SAMPLES"] - executed_samples,
    )


def parse_csim_accuracy(path: Path) -> Optional[float]:
    """Extrai a última acurácia final reportada pelo testbench CSim.

    O log pertence sempre à própria run. Um log ausente, sem a linha final ou
    com percentual fora do intervalo válido é tratado como métrica
    indisponível, sem recorrer a outra execução.
    """
    path = Path(path)
    if not path.is_file():
        return None
    contents = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^\s*\*{3}\s*Final\s+Acc:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*$",
        contents,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        return None
    try:
        accuracy = float(matches[-1])
    except ValueError:
        return None
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 100.0:
        return None
    return accuracy


@dataclass
class PowerReportInfo:
    total_on_chip_power_w: float
    dynamic_power_w: float
    device_static_power_w: float
    total_on_chip_power_display: str
    dynamic_power_display: str
    device_static_power_display: str
    confidence_level: str
    # Ausente no modo vectorless: sem SAIF o Vivado não emite a linha.
    saif_match_percent: Optional[float]
    saif_matched_design_nets: Optional[int]
    saif_total_design_nets: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "total_on_chip_power_w": self.total_on_chip_power_w,
            "dynamic_power_w": self.dynamic_power_w,
            "device_static_power_w": self.device_static_power_w,
            "total_on_chip_power_display": self.total_on_chip_power_display,
            "dynamic_power_display": self.dynamic_power_display,
            "device_static_power_display": self.device_static_power_display,
            "confidence_level": self.confidence_level,
            "saif_match_percent": self.saif_match_percent,
            "saif_matched_design_nets": self.saif_matched_design_nets,
            "saif_total_design_nets": self.saif_total_design_nets,
        }
        if self.saif_matched_design_nets is not None and self.saif_total_design_nets:
            result["saif_match_percent_from_counts"] = (
                100.0
                * self.saif_matched_design_nets
                / self.saif_total_design_nets
            )
        return result


@dataclass
class VivadoUtilizationResourceInfo:
    label: str
    used: float
    available: float
    utilization_percent: float
    recalculated_utilization_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "used": self.used,
            "available": self.available,
            "utilization_percent": self.utilization_percent,
            "recalculated_utilization_percent": self.recalculated_utilization_percent,
        }


@dataclass
class VivadoUtilizationInfo:
    report: Path
    source: str
    design_state: str
    mode: str
    part: str
    device: str
    resources: Dict[str, VivadoUtilizationResourceInfo]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report": str(self.report),
            "source": self.source,
            "design_state": self.design_state,
            "mode": self.mode,
            "part": self.part,
            "device": self.device,
            "resources": {name: resource.to_dict() for name, resource in self.resources.items()},
        }


def _report_table_cell(contents: str, label_pattern: str) -> Optional[str]:
    match = re.search(
        r"^\|\s*{}\s*\|\s*([^|]+?)\s*\|".format(label_pattern),
        contents,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def _parse_power_cell(cell: Optional[str]) -> Optional[Tuple[float, str]]:
    if cell is None:
        return None
    match = re.fullmatch(
        r"\s*(<)?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*",
        cell,
    )
    if match is None:
        return None
    display = "{}{}".format("<" if match.group(1) else "", match.group(2))
    return float(match.group(2)), display


def _normalize_utilization_label(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\*+$", "", value).strip()).lower()


def _parse_utilization_number(cell: Optional[str]) -> Optional[float]:
    if cell is None:
        return None
    text = cell.strip().replace(",", "")
    if not text or text in {"-", "n/a", "na"}:
        return None
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _find_utilization_row(contents: str, label_patterns: Sequence[str]) -> Optional[List[str]]:
    normalized_patterns = {_normalize_utilization_label(pattern) for pattern in label_patterns}
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        label = _normalize_utilization_label(cells[0])
        if label in normalized_patterns:
            return cells
    return None


def _find_utilization_columns(contents: str) -> Optional[Dict[str, int]]:
    """Localiza as colunas pelo cabeçalho do relatório Vivado.

    Vivado 2025.2 inclui a coluna ``Prohibited`` entre ``Fixed`` e
    ``Available``; versões anteriores podem omiti-la. A posição, portanto,
    não pode ser inferida por índices fixos.
    """
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or _normalize_utilization_label(cells[0]) != "site type":
            continue
        normalized = [_normalize_utilization_label(cell) for cell in cells]
        required = {"used", "available", "util%"}
        if required.issubset(normalized):
            return {
                name: normalized.index(name)
                for name in ("used", "available", "util%")
            }
    return None


def parse_vivado_utilization_report(report_path: Path, part: str, mode: str = "out_of_context") -> VivadoUtilizationInfo:
    """Extrai o uso pós-síntese baseado na utilização global do Vivado."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise StageError("Relatório de utilização pós-síntese ausente: {}".format(report_path))
    contents = report_path.read_text(encoding="utf-8", errors="replace")
    device_match = re.search(r"^\|\s*Device\s*:\s*([^\n|]+?)\s*$", contents, flags=re.MULTILINE)
    design_state_match = re.search(
        r"^\|\s*Design\s+State\s*:\s*([^\n|]+?)\s*$",
        contents,
        flags=re.MULTILINE,
    )
    if device_match is None or design_state_match is None:
        raise StageError(
            "Relatório de utilização pós-síntese não contém cabeçalho completo: {}".format(report_path)
        )
    device = device_match.group(1).strip()
    design_state = design_state_match.group(1).strip()
    if device != part:
        raise StageError(
            "Relatório de utilização pós-síntese refere um dispositivo diferente do part configurado: {} != {}".format(
                device,
                part,
            )
        )
    if design_state.lower() != "synthesized":
        raise StageError(
            "Relatório de utilização pós-síntese possui estado inesperado: {}".format(design_state)
        )

    resource_patterns = {
        "lut": ("CLB LUTs", "Slice LUTs"),
        "ff": ("CLB Registers", "Slice Registers"),
        "bram": ("Block RAM Tile",),
        "dsp": ("DSPs", "DSP Blocks"),
        "uram": ("URAM", "UltraRAM"),
    }
    columns = _find_utilization_columns(contents)
    if columns is None:
        raise StageError(
            "Relatório de utilização pós-síntese não contém cabeçalho de recursos "
            "com Used, Available e Util%: {}".format(report_path)
        )
    resources: Dict[str, VivadoUtilizationResourceInfo] = {}
    missing_required: List[str] = []
    for resource_name, patterns in resource_patterns.items():
        row = _find_utilization_row(contents, patterns)
        if row is None:
            if resource_name == "uram":
                continue
            missing_required.append(resource_name)
            continue
        try:
            used = _parse_utilization_number(row[columns["used"]])
            available = _parse_utilization_number(row[columns["available"]])
            utilization_percent = _parse_utilization_number(row[columns["util%"]])
        except IndexError:
            missing_required.append(resource_name)
            continue
        if used is None or available is None or utilization_percent is None:
            missing_required.append(resource_name)
            continue
        if available == 0 and used > 0:
            raise StageError(
                "Relatório de utilização pós-síntese informa capacidade zero para "
                "{} apesar de Used={} (possível erro de parsing): {}".format(
                    resource_name,
                    used,
                    report_path,
                )
            )
        recalculated = 0.0 if available == 0 else 100.0 * used / available
        resources[resource_name] = VivadoUtilizationResourceInfo(
            label=row[0],
            used=used,
            available=available,
            utilization_percent=utilization_percent,
            recalculated_utilization_percent=recalculated,
        )
    if missing_required:
        raise StageError(
            "Relatório de utilização pós-síntese não contém campos obrigatórios ({}): {}".format(
                ", ".join(sorted(missing_required)),
                report_path,
            )
        )
    return VivadoUtilizationInfo(
        report=report_path,
        source="vivado_post_synth",
        design_state=design_state,
        mode=mode,
        part=part,
        device=device,
        resources=resources,
    )


def parse_cosim_report(report_path: Path) -> Dict[str, Any]:
    """Extrai latência, intervalo e tempo total do relatório de co-simulação.

    A tabela traz uma linha por linguagem de RTL e as não executadas aparecem
    com ``NA``; só a linha que reporta um status conclusivo é considerada.  A
    média é o número que interessa num design de latência variável: ela já
    inclui os stalls de contrapressão, que nem a estimativa da síntese nem uma
    contagem de laços em CSim conseguem prever.  ``Total Execution Time`` é o
    tempo do testbench inteiro e é a janela direta para energia, sem depender
    de multiplicar média por número de transações.
    """
    report_path = Path(report_path)
    if not report_path.is_file():
        raise StageError("Relatório de co-simulação ausente: {}".format(report_path))
    contents = report_path.read_text(encoding="utf-8", errors="replace")

    campo = r"\s*([0-9]+|NA)\s*\|"
    padrao = re.compile(
        r"^\|\s*(?P<rtl>[A-Za-z]+)\s*\|\s*(?P<status>[A-Za-z/ ]+?)\s*\|"
        + campo * 7,
        re.MULTILINE,
    )
    linhas = [m for m in padrao.finditer(contents)]
    if not linhas:
        raise StageError(
            "Relatório de co-simulação sem tabela de latência: {}".format(report_path)
        )
    executadas = [m for m in linhas if m.group("status").strip().upper() != "NA"]
    if not executadas:
        raise StageError(
            "Nenhum RTL foi co-simulado neste relatório: {}".format(report_path)
        )
    if len(executadas) > 1:
        raise StageError(
            "Mais de um RTL co-simulado; o relatório é ambíguo: {}".format(report_path)
        )
    linha = executadas[0]

    def numero(indice: int) -> Optional[int]:
        valor = linha.group(indice)
        return None if valor == "NA" else int(valor)

    status = linha.group("status").strip()
    if status.casefold() != "pass":
        raise StageError(
            "Co-simulação não passou (status {!r}): {}".format(status, report_path)
        )
    return {
        "report": str(report_path.resolve()),
        "rtl": linha.group("rtl"),
        "status": status,
        "latency_min_cycles": numero(3),
        "latency_avg_cycles": numero(4),
        "latency_max_cycles": numero(5),
        "interval_min_cycles": numero(6),
        "interval_avg_cycles": numero(7),
        "interval_max_cycles": numero(8),
        "total_execution_cycles": numero(9),
    }


def parse_power_report(report_path: Path) -> PowerReportInfo:
    """Extrai potência, confiança e cobertura da tabela Summary do Vivado."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise StageError("Relatório de potência ausente: {}".format(report_path))
    contents = report_path.read_text(encoding="utf-8", errors="replace")
    total = _parse_power_cell(
        _report_table_cell(contents, r"Total\s+On-Chip\s+Power\s+\(W\)")
    )
    dynamic = _parse_power_cell(
        _report_table_cell(contents, r"Dynamic\s+\(W\)")
    )
    static = _parse_power_cell(
        _report_table_cell(contents, r"Device\s+Static\s+\(W\)")
    )
    confidence = _report_table_cell(contents, r"Confidence\s+Level")
    match_cell = _report_table_cell(contents, r"Design\s+Nets\s+Matched")
    match = (
        re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]+)?)\s*%"
            r"(?:\s*\(\s*([0-9]+)\s*/\s*([0-9]+)\s*\))?\s*",
            match_cell,
        )
        if match_cell is not None
        else None
    )
    missing = []
    for name, value in (
        ("Total On-Chip Power", total),
        ("Dynamic", dynamic),
        ("Device Static", static),
        ("Confidence Level", confidence),
    ):
        if value is None:
            missing.append(name)
    if missing:
        raise StageError(
            "Relatório de potência não contém campos obrigatórios ({}): {}".format(
                ", ".join(missing), report_path
            )
        )
    assert total is not None
    assert dynamic is not None
    assert static is not None
    assert confidence is not None
    return PowerReportInfo(
        total_on_chip_power_w=total[0],
        dynamic_power_w=dynamic[0],
        device_static_power_w=static[0],
        total_on_chip_power_display=total[1],
        dynamic_power_display=dynamic[1],
        device_static_power_display=static[1],
        confidence_level=confidence,
        saif_match_percent=float(match.group(1)) if match is not None else None,
        saif_matched_design_nets=(
            int(match.group(2)) if match is not None and match.group(2) else None
        ),
        saif_total_design_nets=(
            int(match.group(3)) if match is not None and match.group(3) else None
        ),
    )


def _yaml_or_json_write(path: Path, payload: Dict[str, Any]) -> None:
    """JSON também é YAML válido e evita tornar a criação de runs dependente de PyYAML."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _yaml_or_json_read(path: Path) -> Dict[str, Any]:
    try:
        return read_json(path)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise ValidationError("Não foi possível ler o manifesto YAML sem PyYAML") from error
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValidationError("Manifesto de execução inválido: {}".format(path))
        return payload


def parse_saif(path: Path) -> SaifInfo:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise StageError("SAIF ausente ou vazio: {}".format(path))
    contents = path.read_text(encoding="utf-8", errors="replace")
    duration_match = re.search(r"\(DURATION\s+([0-9]+(?:\.[0-9]+)?)\s*\)", contents)
    if duration_match is None:
        raise StageError("SAIF não contém DURATION: {}".format(path))
    duration = float(duration_match.group(1))
    if duration <= 0:
        raise StageError("SAIF possui DURATION não positiva: {}".format(path))
    timescale_match = re.search(r"\(TIMESCALE\s+([^\)]+)\)", contents)
    transitions = [int(value) for value in re.findall(r"\(TC\s+([0-9]+)\s*\)", contents)]
    transition_count = sum(transitions)
    if transition_count <= 0:
        raise StageError("SAIF não contém transições (TC) positivas: {}".format(path))
    return SaifInfo(
        path=path,
        size_bytes=path.stat().st_size,
        duration=duration,
        timescale=timescale_match.group(1).strip() if timescale_match else None,
        transition_count=transition_count,
    )


def parse_saif_match(report_path: Path) -> Optional[float]:
    if not report_path.is_file():
        return None
    contents = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Design\s+Nets\s+Matched\s*\|\s*([0-9]+(?:\.[0-9]+)?)%",
        contents,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def derive_summary_metrics(
    workload: Optional[Dict[str, Any]],
    activity: Optional[Dict[str, Any]],
    power: Optional[Dict[str, Any]],
    analytic_duration_seconds: Optional[float] = None,
    analytic_duration_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Calcula métricas normalizadas usando uma única janela de execução.

    Com SAIF, a janela é a captura medida na simulação pós-síntese.  Sem SAIF
    não existe captura, e ``analytic_duration_seconds`` fornece a janela a
    partir da latência relatada pelo HLS.  As duas origens são registradas em
    ``latency_definition`` e ``energy_definition``, porque a energia é sempre
    potência média vezes a janela e o resultado só é interpretável junto com a
    origem dela.
    """
    metrics: Dict[str, Any] = {
        "latency_definition": "full_saif_window_amortized",
        "energy_definition": "average_power_times_full_saif_window",
    }
    duration_seconds: Any = None
    if activity:
        duration_seconds = activity.get("duration_seconds")
        try:
            duration_seconds = float(duration_seconds)
        except (TypeError, ValueError):
            duration_seconds = saif_duration_seconds(
                activity.get("duration"), activity.get("timescale")
            )
    if (
        duration_seconds is None
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ) and analytic_duration_seconds is not None:
        duration_seconds = analytic_duration_seconds
        metrics["latency_definition"] = analytic_duration_label or "hls_latency_times_logical_steps"
        metrics["energy_definition"] = "average_power_times_{}".format(
            metrics["latency_definition"]
        )
    if (
        duration_seconds is None
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        metrics["unavailable_reason"] = (
            "sem janela de execução: TIMESCALE SAIF ausente ou inválida e "
            "latência HLS indisponível"
        )
        return metrics
    metrics["capture_duration_seconds"] = duration_seconds

    executed_samples = 0
    total_steps = 0
    if workload:
        try:
            executed_samples = int(workload.get("executed_samples", 0))
            total_steps = int(workload.get("total_logical_steps", 0))
        except (TypeError, ValueError):
            executed_samples = 0
            total_steps = 0
    if executed_samples > 0:
        metrics["average_latency_per_sample_seconds"] = (
            duration_seconds / executed_samples
        )
    if total_steps > 0:
        metrics["average_latency_per_step_seconds"] = (
            duration_seconds / total_steps
        )

    if not power:
        return metrics
    power_fields = {
        "total": "total_on_chip_power_w",
        "dynamic": "dynamic_power_w",
        "static": "device_static_power_w",
    }
    parsed_power: Dict[str, float] = {}
    for kind, field_name in power_fields.items():
        try:
            value = float(power[field_name])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            parsed_power[kind] = value
            metrics["capture_energy_{}_joules".format(kind)] = (
                value * duration_seconds
            )
            if executed_samples > 0:
                metrics["energy_per_sample_{}_joules".format(kind)] = (
                    value * duration_seconds / executed_samples
                )
            if total_steps > 0:
                metrics["energy_per_step_{}_joules".format(kind)] = (
                    value * duration_seconds / total_steps
                )
    metrics["power_values_available"] = sorted(parsed_power)
    metrics["power_metrics_provisional"] = (
        not bool(power.get("saif_coverage_passed", False))
        or bool(power.get("capture_is_partial", False))
    )
    return metrics


def _format_pt_decimal(value: Any, digits: int) -> str:
    return ("{:.%df}" % digits).format(float(value)).replace(".", ",")


def _format_pt_integer(value: Any) -> str:
    return "{:,}".format(int(value)).replace(",", ".")


def _format_power_display(power: Dict[str, Any], name: str) -> str:
    display = power.get(name + "_display")
    if display is not None:
        return str(display).replace(".", ",")
    return _format_pt_decimal(power[name + "_w"], 3)


def _format_resource_quantity(value: Any, fractional_digits: int = 0) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(numeric) and numeric.is_integer():
        return _format_pt_integer(int(numeric))
    return _format_pt_decimal(numeric, fractional_digits)


def _format_resource_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return _format_pt_decimal(numeric, 3)


def _format_saif_duration_expression(activity: Dict[str, Any]) -> str:
    duration = activity.get("duration")
    try:
        numeric_duration = float(duration)
        rendered_duration = (
            _format_pt_integer(numeric_duration)
            if numeric_duration.is_integer()
            else _format_pt_decimal(numeric_duration, 9)
        )
    except (TypeError, ValueError):
        rendered_duration = str(duration)
    timescale = str(activity.get("timescale") or "unidade desconhecida").strip()
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*(s|ms|us|µs|ns|ps|fs)",
        timescale,
        flags=re.IGNORECASE,
    )
    if match and float(match.group(1)) == 1.0:
        return "{} {}".format(rendered_duration, match.group(2))
    return "{} × {}".format(rendered_duration, timescale)


class Pipeline:
    """Uma execução isolada do fluxo de simulação."""

    def __init__(
        self,
        run_dir: Path,
        project_info: ProjectInfo,
        config: RunConfig,
        dry_run: bool = False,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.project_info = project_info
        self.config = config.validate()
        self.dry_run = dry_run
        self.project_dir = self.run_dir / "project"
        self.logs_dir = self.run_dir / "logs"
        self.reports_dir = self.run_dir / "reports"
        self.artifacts_path = self.run_dir / "artifacts.json"
        self.status_path = self.run_dir / "status.json"
        self.manifest_path = self.run_dir / "run.yaml"
        self.runner = CommandRunner(
            timeout_seconds=self.config.timeout_seconds,
            dry_run=dry_run,
        )

    @classmethod
    def create(
        cls,
        project_path: Path,
        config: RunConfig,
        runs_root: Path,
        dry_run: bool = False,
    ) -> "Pipeline":
        runs_root = Path(runs_root).expanduser().resolve()
        project_info = validate_project(project_path, runs_root=runs_root)
        component_dir = runs_root / project_info.identifier
        run_id = "{}-{}".format(run_stamp(), project_info.project_hash[:12])
        run_dir = component_dir / run_id
        if run_dir.exists():
            raise ValidationError(
                "Já existe uma execução com as mesmas entradas: {}. Use resume.".format(run_dir)
            )
        for directory in (
            run_dir,
            run_dir / "logs",
            run_dir / "reports",
            run_dir / "10_neurohls",
            run_dir / "20_hls",
            run_dir / "30_export",
            run_dir / "40_vivado_synth",
            run_dir / "50_post_synth_sim",
            run_dir / "60_activity",
            run_dir / "70_power",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        pipeline = cls(run_dir, project_info, config, dry_run=dry_run)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": utc_now(),
            "source_project": str(project_info.source_root),
            "project": project_info.to_dict(),
            "configuration": config.to_dict(),
        }
        _yaml_or_json_write(pipeline.manifest_path, manifest)
        atomic_write_json(
            pipeline.status_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": utc_now(),
                "state": "created",
                "stages": {},
            },
        )
        atomic_write_json(pipeline.artifacts_path, {})
        return pipeline

    @classmethod
    def load(cls, run_dir: Path, dry_run: bool = False) -> "Pipeline":
        run_dir = Path(run_dir).expanduser().resolve()
        manifest_path = run_dir / "run.yaml"
        if not manifest_path.is_file():
            raise ValidationError("Manifesto de execução ausente: {}".format(manifest_path))
        manifest = _yaml_or_json_read(manifest_path)
        try:
            project_info = ProjectInfo.from_dict(manifest["project"])
            config = RunConfig.from_dict(manifest["configuration"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError("Manifesto de execução inválido: {}".format(manifest_path)) from error
        return cls(run_dir, project_info, config, dry_run=dry_run)

    @property
    def hls_solution_dir(self) -> Path:
        return self.project_dir / self.config.hls_project_name / self.config.hls_solution_name

    def status(self) -> Dict[str, Any]:
        if not self.status_path.is_file():
            raise ValidationError("Status inexistente: {}".format(self.status_path))
        return read_json(self.status_path)

    def artifacts(self) -> Dict[str, Any]:
        if not self.artifacts_path.is_file():
            return {}
        return read_json(self.artifacts_path)

    def _save_status(self, status: Dict[str, Any]) -> None:
        atomic_write_json(self.status_path, status)

    def _set_artifact(self, name: str, value: Any) -> None:
        artifacts = self.artifacts()
        artifacts[name] = value
        atomic_write_json(self.artifacts_path, artifacts)

    def _stage_output_paths(self, stage: str) -> List[Path]:
        outputs = {
            "prepare": [self.project_dir / "snn_implementation.cpp"],
            "vitis-project": [self.project_dir / self.config.hls_project_name],
            "csim": [self.logs_dir / "csim.log"],
            "hls-synth": [self.hls_solution_dir / "impl" / "verilog"],
            "cosim-setup": [self.hls_solution_dir / "sim" / "verilog" / "{}.prj".format(self.project_info.top)],
            "cosim": [self.reports_dir / "cosim_summary.json"],
            "export": [self.run_dir / "30_export" / "export.json"],
            "vivado-synth": [
                self.run_dir / "40_vivado_synth" / "post_synth.dcp",
                self.run_dir / "40_vivado_synth" / "post_synth_netlist.v",
                self.run_dir / "40_vivado_synth" / "utilization_post_synth.rpt",
                self.run_dir / "40_vivado_synth" / "utilization_device_post_synth.rpt",
                self.reports_dir / "utilization_summary.json",
                self.run_dir / "40_vivado_synth" / "timing_post_synth.rpt",
                self.run_dir / "40_vivado_synth" / "methodology_post_synth.rpt",
            ],
            "post-synth-sim": [self.run_dir / "60_activity" / "post_synth_activity.saif"],
            "power": [self.run_dir / "70_power" / "power_report.rpt"],
        }
        return outputs[stage]

    def _stage_is_required(self, stage: str) -> bool:
        """Falso quando a configuração torna a etapa desnecessária."""
        # post-synth-sim existe apenas para produzir o SAIF.
        if stage == "post-synth-sim":
            return self.config.activity_source != "vectorless"
        # cosim-setup prepara os vetores de RTL, consumidos tanto pela
        # co-simulação quanto pela simulação pós-síntese.
        if stage == "cosim-setup":
            return self.config.activity_source != "vectorless" or self.config.run_cosim
        if stage == "cosim":
            return self.config.run_cosim
        return True

    def _stage_is_reusable(self, stage: str) -> bool:
        entry = self.status().get("stages", {}).get(stage, {})
        return entry.get("state") in ("success", "skipped") and all(
            path.exists() and (not path.is_file() or path.stat().st_size > 0)
            for path in self._stage_output_paths(stage)
        )

    def run(
        self,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
        force: bool = False,
    ) -> None:
        from_stage = from_stage or STAGE_ORDER[0]
        to_stage = to_stage or STAGE_ORDER[-1]
        if from_stage not in STAGE_ORDER:
            raise ValidationError("Etapa inicial inválida: {}".format(from_stage))
        if to_stage not in STAGE_ORDER:
            raise ValidationError("Etapa final inválida: {}".format(to_stage))
        start = STAGE_ORDER.index(from_stage)
        end = STAGE_ORDER.index(to_stage)
        if start > end:
            raise ValidationError("A etapa inicial deve preceder a etapa final")
        if start > 0:
            for predecessor in STAGE_ORDER[:start]:
                if not self._stage_is_required(predecessor):
                    continue
                if not self._stage_is_reusable(predecessor):
                    raise ValidationError(
                        "Não é possível iniciar em {}: a etapa anterior {} não está válida nesta run"
                        .format(from_stage, predecessor)
                    )

        status = self.status()
        status["execution_scope"] = {
            "from_stage": from_stage,
            "to_stage": to_stage,
        }
        self._save_status(status)

        if self.dry_run:
            for stage in STAGE_ORDER[start : end + 1]:
                self._mark_stage(stage, "planned", reason="dry-run: ferramenta não executada")
            status = self.status()
            status["state"] = "dry-run"
            status["finished_at"] = utc_now()
            self._save_status(status)
            self._write_summary()
            return

        try:
            for stage in STAGE_ORDER[start : end + 1]:
                if not self._stage_is_required(stage):
                    self._mark_stage(
                        stage, "skipped",
                        reason="não requerida por power.activity_source='{}'".format(
                            self.config.activity_source
                        ),
                    )
                    continue
                if not force and self.config.reuse_cache and self._stage_is_reusable(stage):
                    self._mark_stage(stage, "skipped", reason="artefatos válidos reutilizados")
                    continue
                self._run_stage(stage)
        except KeyboardInterrupt:
            try:
                self._write_summary()
            except Exception:
                pass
            raise
        except Exception:
            # Preserve o erro original, mas mantenha os relatórios coerentes com
            # status.json mesmo quando uma ferramenta interromper a execução.
            try:
                self._write_summary()
            except Exception:
                pass
            raise
        status = self.status()
        status["state"] = "success"
        status["finished_at"] = utc_now()
        self._save_status(status)
        self._write_summary()

    def _mark_stage(self, stage: str, state: str, **fields: Any) -> None:
        status = self.status()
        status["state"] = state if state == "failed" else "running"
        if state in ("running", "skipped", "planned"):
            status.pop("finished_at", None)
        elif state == "failed":
            fields.setdefault("finished_at", utc_now())
            status["finished_at"] = fields["finished_at"]
        entry = status.setdefault("stages", {}).setdefault(stage, {})
        stale_fields = {
            "running": ("error", "reason", "finished_at"),
            "success": ("error", "reason"),
            "skipped": ("error", "started_at", "finished_at"),
            "planned": ("error", "started_at", "finished_at"),
            "failed": ("reason",),
        }
        for stale_field in stale_fields.get(state, ()):
            entry.pop(stale_field, None)
        entry.update(fields)
        entry["state"] = state
        entry["updated_at"] = utc_now()
        self._save_status(status)

    def _run_stage(self, stage: str) -> None:
        handler = getattr(self, "_stage_" + stage.replace("-", "_"))
        self._mark_stage(stage, "running", started_at=utc_now())
        try:
            handler()
            if not self.dry_run:
                missing = [
                    str(path)
                    for path in self._stage_output_paths(stage)
                    if not path.exists() or (path.is_file() and path.stat().st_size == 0)
                ]
                if missing:
                    raise StageError(
                        "A etapa {} terminou sem artefatos obrigatórios: {}".format(
                            stage, ", ".join(missing)
                        )
                    )
            self._mark_stage(stage, "success", finished_at=utc_now())
        except KeyboardInterrupt:
            self._mark_stage(
                stage,
                "failed",
                finished_at=utc_now(),
                error="Execução interrompida pelo usuário",
            )
            raise
        except Exception as error:
            self._mark_stage(stage, "failed", finished_at=utc_now(), error=str(error))
            raise

    def _tool_command(self, command: Sequence[str]) -> Tuple[List[str], Optional[Dict[str, str]]]:
        settings = self.config.settings_script
        if not settings:
            return [str(item) for item in command], None
        settings_path = Path(settings).expanduser().resolve()
        if not settings_path.is_file():
            raise ValidationError("tools.settings_script inexistente: {}".format(settings_path))
        shell_command = "source {} && exec {}".format(
            shlex.quote(str(settings_path)), command_text([str(item) for item in command])
        )
        return ["bash", "-lc", shell_command], None

    def _run_tool(self, stage: str, command: Sequence[str], cwd: Path, suffix: str = "") -> None:
        full_command, environment = self._tool_command(command)
        log_name = stage + ("-" + suffix if suffix else "") + ".log"
        self.runner.run(full_command, cwd=cwd, log_path=self.logs_dir / log_name, environment=environment)

    def _write_wrapper(self, name: str, contents: str) -> Path:
        wrapper = self.project_dir / ".sim_wrappers" / name
        atomic_write_text(wrapper, contents)
        return wrapper

    def _stage_prepare(self) -> None:
        if self.project_dir.exists():
            if self.dry_run:
                return
            raise StageError("Diretório de trabalho já existe: {}".format(self.project_dir))
        copy_project(self.project_info.source_root, self.project_dir)
        copied_info = validate_project(self.project_dir)
        if copied_info.project_hash != self.project_info.project_hash:
            raise StageError("A cópia do projeto não preservou o hash da entrada")
        inventory = {
            "source_project": str(self.project_info.source_root),
            "project_hash": self.project_info.project_hash,
            "top": self.project_info.top,
            "files": [
                {
                    "path": str(path.relative_to(self.project_dir)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(self.project_dir.rglob("*"))
                if path.is_file()
            ],
        }
        atomic_write_json(self.run_dir / "10_neurohls" / "project_inventory.json", inventory)
        self._set_artifact("project_copy", str(self.project_dir))

    def _stage_vitis_project(self) -> None:
        wrapper = self._write_wrapper(
            "create_project.tcl",
            vitis_wrapper(self.project_dir / "0_create_project.tcl", [self.config.hls_project_name]),
        )
        self._run_tool(
            "vitis-project",
            [self.config.vitis_run, "--mode", "hls", "--tcl", str(wrapper)],
            self.project_dir,
        )
        if not self.dry_run and not (self.project_dir / self.config.hls_project_name).is_dir():
            raise StageError("Vitis não criou o projeto {}".format(self.config.hls_project_name))

    def _stage_csim(self) -> None:
        wrapper = self._write_wrapper(
            "csim.tcl",
            vitis_wrapper(
                self.project_dir / "1_csim.tcl",
                [self.config.hls_project_name, self.config.hls_solution_name],
            ),
        )
        self._run_tool(
            "csim",
            [self.config.vitis_run, "--mode", "hls", "--tcl", str(wrapper)],
            self.project_dir,
        )

    def _activate_medoid_testbench(self) -> bool:
        """Troca apenas a cópia do run para o workload reduzido de RTL."""
        sources = [self.project_dir / relative for relative in MEDOID_TESTBENCH_FILES]
        if not all(path.is_file() for path in sources):
            return False

        destinations = (
            self.project_dir / "testbench.cpp",
            self.project_dir / "tb_data" / "data.txt",
            self.project_dir / "tb_data" / "targets.txt",
        )
        for source, destination in zip(sources, destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        workload = parse_testbench_workload(destinations[0])
        details = {
            "profile": "medoids",
            "testbench": str(destinations[0]),
            "data": str(destinations[1]),
            "targets": str(destinations[2]),
        }
        if workload is not None:
            details["workload"] = workload.to_dict()
        self._set_artifact("rtl_simulation_testbench", details)
        return True

    def _stage_hls_synth(self) -> None:
        wrapper = self._write_wrapper(
            "hls_synth.tcl",
            vitis_wrapper(
                self.project_dir / "2_synth.tcl",
                [
                    self.config.hls_project_name,
                    self.config.hls_solution_name,
                    "{:.12g}".format(self.config.clock_period_ns),
                    self.config.part,
                ],
            ),
        )
        self._run_tool(
            "hls-synth",
            [self.config.vitis_run, "--mode", "hls", "--tcl", str(wrapper)],
            self.project_dir,
        )
        rtl_dir = self.hls_solution_dir / "impl" / "verilog"
        if not self.dry_run and not any(rtl_dir.glob("*.v")):
            raise StageError("Síntese HLS não produziu RTL em {}".format(rtl_dir))
        self._set_artifact("hls_rtl_dir", str(rtl_dir))

    def _stage_cosim_setup(self) -> None:
        # CSim já foi executado com o dataset integral. A partir deste ponto,
        # CoSim e o AutoTB reutilizado pela simulação pós-síntese usam somente
        # os medoids, quando o projeto fornece o bundle tb_medoids/.
        self._activate_medoid_testbench()
        wrapper = self._write_wrapper(
            "cosim_setup.tcl",
            cosim_setup_tcl(self.config.hls_project_name, self.config.hls_solution_name),
        )
        self._run_tool(
            "cosim-setup",
            [self.config.vitis_run, "--mode", "hls", "--tcl", str(wrapper)],
            self.project_dir,
        )

    def _stage_cosim(self) -> None:
        wrapper = self._write_wrapper(
            "cosim.tcl",
            cosim_tcl(self.config.hls_project_name, self.config.hls_solution_name),
        )
        self._run_tool(
            "cosim",
            [self.config.vitis_run, "--mode", "hls", "--tcl", str(wrapper)],
            self.project_dir,
        )
        info = parse_cosim_report(
            self.hls_solution_dir / "sim" / "report" / "{}_cosim.rpt".format(
                self.project_info.top
            )
        )
        atomic_write_json(self.reports_dir / "cosim_summary.json", info)
        self._set_artifact("cosim", info)

    def _stage_export(self) -> None:
        wrapper = self._write_wrapper(
            "export.tcl",
            export_tcl(self.config.hls_project_name, self.config.hls_solution_name),
        )
        self._run_tool(
            "export",
            [self.config.vitis_run, "--mode", "hls", "--tcl", str(wrapper)],
            self.project_dir,
        )
        rtl_dir = self._find_exported_rtl()
        export_zip = self.hls_solution_dir / "impl" / "export.zip"
        export_info = {
            "rtl_dir": str(rtl_dir),
            "rtl_files": [str(path) for path in sorted(rtl_dir.rglob("*.v"))],
            "export_zip": str(export_zip) if export_zip.is_file() else None,
            "xdc_files": [str(path) for path in self._find_xdc_files()],
        }
        atomic_write_json(self.run_dir / "30_export" / "export.json", export_info)
        self._set_artifact("export", export_info)

    def _find_exported_rtl(self) -> Path:
        candidates = (
            self.hls_solution_dir / "impl" / "ip" / "hdl" / "verilog",
            self.hls_solution_dir / "impl" / "verilog",
        )
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.rglob("*.v")):
                return candidate.resolve()
        raise StageError(
            "Não foi possível localizar RTL exportado em {}".format(self.hls_solution_dir / "impl")
        )

    def _find_xdc_files(self) -> List[Path]:
        impl_dir = self.hls_solution_dir / "impl"
        return sorted(path.resolve() for path in impl_dir.rglob("*.xdc") if path.is_file())

    def _stage_vivado_synth(self) -> None:
        synth_dir = self.run_dir / "40_vivado_synth"
        rtl_dir = self._find_exported_rtl()
        clock_xdc = synth_dir / "clock_override.xdc"
        dcp_path = synth_dir / "post_synth.dcp"
        netlist_path = synth_dir / "post_synth_netlist.v"
        util_path = synth_dir / "utilization_post_synth.rpt"
        device_util_path = synth_dir / "utilization_device_post_synth.rpt"
        timing_path = synth_dir / "timing_post_synth.rpt"
        methodology_path = synth_dir / "methodology_post_synth.rpt"
        utilization_summary_path = self.reports_dir / "utilization_summary.json"
        script = synth_dir / "synth_ooc.tcl"
        atomic_write_text(clock_xdc, clock_override_xdc(self.config))
        xdc_files = self._find_xdc_files()
        xdc_files.append(clock_xdc)
        atomic_write_text(
            script,
            vivado_synth_tcl(
                self.config,
                self.project_info.top,
                rtl_dir,
                xdc_files,
                dcp_path,
                netlist_path,
                util_path,
                device_util_path,
                timing_path,
                methodology_path,
            ),
        )
        self._run_tool(
            "vivado-synth",
            [self.config.vivado, "-mode", "batch", "-source", str(script)],
            synth_dir,
        )
        utilization_summary = parse_vivado_utilization_report(
            device_util_path,
            self.config.part,
            mode="out_of_context",
        ).to_dict()
        atomic_write_json(utilization_summary_path, utilization_summary)
        self._set_artifact(
            "vivado_synth",
            {
                "checkpoint": str(dcp_path),
                "netlist": str(netlist_path),
                "utilization": str(util_path),
                "utilization_hierarchical": str(util_path),
                "utilization_device": str(device_util_path),
                "timing": str(timing_path),
                "methodology": str(methodology_path),
                "rtl_dir": str(rtl_dir),
                "clock_constraints": str(clock_xdc),
                "utilization_summary": utilization_summary,
            },
        )

    def _autotb_metadata(self, autotb_path: Path) -> Tuple[str, str]:
        contents = autotb_path.read_text(encoding="utf-8", errors="replace")
        top_match = re.search(r"^\s*`define\s+AUTOTB_TOP\s+([A-Za-z_][A-Za-z0-9_]*)", contents, re.MULTILINE)
        instance_match = re.search(
            r"^\s*`define\s+AUTOTB_DUT_INST\s+([A-Za-z_][A-Za-z0-9_]*)",
            contents,
            re.MULTILINE,
        )
        if not top_match or not instance_match:
            raise StageError("Não foi possível identificar top e instância no autotestbench {}".format(autotb_path))
        return top_match.group(1), instance_match.group(1)

    def _strip_dataflow_monitor(self, source: Path, destination: Path) -> None:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines(True)
        result: List[str] = []
        skipping = False
        for line in lines:
            if not skipping and "dataflow status monitor" in line.lower():
                skipping = True
                continue
            if skipping:
                if line.strip() == "endmodule":
                    result.append(line)
                    skipping = False
                continue
            result.append(line)
        if skipping:
            raise StageError("Bloco de monitor de dataflow não terminou em {}".format(source))
        atomic_write_text(destination, "".join(result))

    def _post_synth_prj(
        self,
        source_prj: Path,
        output_prj: Path,
        post_autotb: Path,
        netlist: Path,
        stubs: Path,
    ) -> None:
        top = self.project_info.top
        netlist_defines_glbl = False
        if netlist.is_file():
            netlist_contents = netlist.read_text(encoding="utf-8", errors="replace")
            netlist_defines_glbl = bool(
                re.search(r"^\s*module\s+glbl\b", netlist_contents, re.MULTILINE)
            )
        selected: List[str] = []
        for line in source_prj.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                selected.append(line)
                continue
            quoted = re.search(r'"([^"]+)"', line)
            candidate = quoted.group(1) if quoted else ""
            normalized = candidate.replace("\\", "/")
            path_parts = PurePosixPath(normalized).parts
            basename = path_parts[-1] if path_parts else ""
            if basename == "{}.autotb.v".format(top):
                selected.append('sv xil_defaultlib "{}"'.format(post_autotb.resolve()))
                continue
            if basename == "glbl.v" and netlist_defines_glbl:
                continue
            if basename == "dataflow_monitor.sv" or basename.startswith("AESL_deadlock_"):
                continue
            if basename.startswith(top + ".") or basename.startswith(top + "_"):
                if "{}_subsystem".format(top) not in path_parts:
                    continue
            if quoted and candidate:
                source_path = Path(candidate)
                if not source_path.is_absolute():
                    source_path = (source_prj.parent / source_path).resolve()
                if not source_path.is_file():
                    raise StageError(
                        "Fonte referenciada pelo projeto de co-simulação não existe: {}"
                        .format(source_path)
                    )
                line = "{}{}{}".format(
                    line[: quoted.start(1)],
                    source_path,
                    line[quoted.end(1) :],
                )
            selected.append(line)

        insert_at = 1 if selected else 0
        selected.insert(insert_at, 'sv xil_defaultlib "{}"'.format(netlist.resolve()))
        selected.append('sv xil_defaultlib "{}"'.format(stubs.resolve()))
        atomic_write_text(output_prj, "\n".join(selected) + "\n")

    def _xelab_extra_arguments(self, sim_dir: Path) -> List[str]:
        """Reaproveita bibliotecas da execução HLS quando o script existe."""
        script = sim_dir / "run_xsim.sh"
        extras: List[str] = []
        if script.is_file():
            for line in script.read_text(encoding="utf-8", errors="replace").splitlines():
                if "xelab" not in line:
                    continue
                try:
                    tokens = shlex.split(line)
                except ValueError:
                    continue
                for index, token in enumerate(tokens[:-1]):
                    if token in ("-L", "--lib", "-i"):
                        value = tokens[index + 1]
                        pair = [token, value]
                        if pair not in [extras[position : position + 2] for position in range(0, len(extras) - 1)]:
                            extras.extend(pair)
                if extras:
                    return extras
        return [
            "-L", "xil_defaultlib",
            "-L", "unisims_ver",
            "-L", "xpm",
            "-L", "floating_point_v7_1_21",
            "-L", "floating_point_v7_0_26",
            "-L", "uvm",
        ]

    def _stage_post_synth_sim(self) -> None:
        sim_dir = self.hls_solution_dir / "sim" / "verilog"
        source_prj = sim_dir / "{}.prj".format(self.project_info.top)
        source_autotb = sim_dir / "{}.autotb.v".format(self.project_info.top)
        netlist = self.run_dir / "40_vivado_synth" / "post_synth_netlist.v"
        if not source_prj.is_file() or not source_autotb.is_file():
            raise StageError("Co-sim setup não produziu .prj/autotb em {}".format(sim_dir))
        if not netlist.is_file():
            raise StageError("Netlist pós-síntese ausente: {}".format(netlist))

        post_dir = self.run_dir / "50_post_synth_sim"
        post_autotb = post_dir / "post_synth.autotb.v"
        post_prj = post_dir / "post_synth.prj"
        stubs = post_dir / "post_synth_monitor_stubs.sv"
        saif = self.run_dir / "60_activity" / "post_synth_activity.saif"
        sim_tcl = post_dir / "capture_saif.tcl"
        self._strip_dataflow_monitor(source_autotb, post_autotb)
        # The post-synthesis testbench normally executes the complete workload.
        # For constrained CI/debug environments an explicit transaction limit
        # can be supplied without changing the generated project:
        # ``SIM_POST_SYNTH_TRANSACTION_LIMIT=<N>``.  This is intentionally
        # opt-in; normal executions keep the complete workload and therefore
        # retain the original timing/energy semantics.
        transaction_limit = self._post_synth_transaction_limit()
        if transaction_limit is not None:
            self._limit_post_synth_transactions(sim_dir, post_autotb, transaction_limit)
        atomic_write_text(
            stubs,
            """`timescale 1ns/1ps
module AESL_deadlock_detector(input dl_reset, input all_finish, input dl_clock); endmodule
module dataflow_monitor(input clock, input reset, input finish); endmodule
""",
        )
        self._post_synth_prj(source_prj, post_prj, post_autotb, netlist, stubs)
        autotb_top, dut_instance = self._autotb_metadata(source_autotb)
        atomic_write_text(sim_tcl, xsim_saif_tcl(saif, autotb_top, dut_instance))

        executable = "post_synth_sim"
        xelab = [
            self.config.xelab,
            "xil_defaultlib.{}".format(autotb_top),
            "xil_defaultlib.glbl",
            "-Oenable_linking_all_libraries",
            "-d", "POST_SYN",
            "-prj", str(post_prj.resolve()),
        ]
        xelab.extend(self._xelab_extra_arguments(sim_dir))
        xelab.extend(["-relax", "-s", executable, "-debug", "all"])
        self._run_tool("post-synth-sim", xelab, sim_dir, suffix="xelab")
        xsim = [
            self.config.xsim,
            "-testplusarg", "UVM_VERBOSITY=UVM_NONE",
            "-testplusarg", "UVM_TESTNAME={}_test_lib".format(self.project_info.top),
            "-testplusarg", "UVM_TIMEOUT=20000000000000",
            "--noieeewarnings",
            executable,
            "-tclbatch", str(sim_tcl.resolve()),
        ]
        self._run_tool("post-synth-sim", xsim, sim_dir, suffix="xsim")
        xsim_log = self.logs_dir / "post-synth-sim-xsim.log"
        xsim_text = xsim_log.read_text(encoding="utf-8", errors="replace")
        failure_markers = ("UVM_FATAL", "ERROR: Simulation using HLS TB failed", "Simulation failed")
        if any(marker in xsim_text for marker in failure_markers):
            raise StageError("Testbench pós-síntese reportou erro; consulte {}".format(xsim_log))
        info = parse_saif(saif)
        activity = info.to_dict()
        activity.update(
            {
                "autotb_top": autotb_top,
                "dut_instance": dut_instance,
                "strip_path": "{}/{}".format(autotb_top, dut_instance),
                "capture_scope": self.config.capture_scope,
                "warmup_samples": self.config.warmup_samples,
                "measured_samples": self.config.measured_samples,
            }
        )
        if transaction_limit is not None:
            activity.update(
                {
                    "transaction_limit": transaction_limit,
                    "capture_is_partial": True,
                    "capture_limit_reason": "SIM_POST_SYNTH_TRANSACTION_LIMIT",
                }
            )
        workload = self._workload_summary()
        if workload is not None:
            activity["workload"] = workload
        atomic_write_json(self.reports_dir / "activity_summary.json", activity)
        self._set_artifact("activity", activity)

    @staticmethod
    def _post_synth_transaction_limit() -> Optional[int]:
        """Return an explicitly requested post-synthesis transaction limit.

        The default is ``None`` so production runs always execute every
        transaction generated by Vitis HLS.  A small positive limit is useful
        when validating the Vivado/XSim plumbing on machines where a complete
        post-synthesis workload is prohibitively slow.
        """
        raw = os.environ.get("SIM_POST_SYNTH_TRANSACTION_LIMIT", "").strip()
        if not raw:
            return None
        try:
            value = int(raw, 10)
        except ValueError as exc:
            raise StageError(
                "SIM_POST_SYNTH_TRANSACTION_LIMIT deve ser um inteiro positivo"
            ) from exc
        if value <= 0:
            raise StageError("SIM_POST_SYNTH_TRANSACTION_LIMIT deve ser maior que zero")
        return value

    @staticmethod
    def _limit_post_synth_transactions(
        sim_dir: Path, post_autotb: Path, transaction_limit: int
    ) -> None:
        """Limit only the copied post-synthesis testbench workload.

        HLS emits the transaction count as a literal in the generated UVM
        sequence/reference model and in the AutoTB performance monitor.  The
        source project is already a per-run copy, so replacing that literal in
        the copy is safe and leaves future full runs unchanged.
        """
        candidates = list((sim_dir / "svtb").rglob("*.sv")) + list(
            (sim_dir / "snn_to_hls_subsystem").rglob("*.sv")
        )
        old_count = re.compile(r"(?<![A-Za-z0-9_])1792(?![A-Za-z0-9_])")
        replacement = str(transaction_limit)
        for source in candidates:
            text = source.read_text(encoding="utf-8", errors="replace")
            if "1792" not in text:
                continue
            updated = old_count.sub(replacement, text)
            if updated != text:
                atomic_write_text(source, updated)

        autotb_text = post_autotb.read_text(encoding="utf-8", errors="replace")
        autotb_text, count = re.subn(
            r"(parameter\s+AUTOTB_TRANSACTION_NUM\s*=\s*)\d+(\s*;)",
            r"\g<1>{}\2".format(transaction_limit),
            autotb_text,
            count=1,
        )
        if count != 1:
            raise StageError(
                "Não foi possível limitar AUTOTB_TRANSACTION_NUM no testbench pós-síntese"
            )
        atomic_write_text(post_autotb, autotb_text)

    def _stage_power(self) -> None:
        dcp = self.run_dir / "40_vivado_synth" / "post_synth.dcp"
        power_dir = self.run_dir / "70_power"
        report = power_dir / "power_report.rpt"
        vectorless = self.config.activity_source == "vectorless"

        if vectorless:
            saif: Optional[Path] = None
            strip_path: Optional[str] = None
            unmatched: Optional[Path] = None
            script = power_dir / "power_vectorless.tcl"
        else:
            activity = self.artifacts().get("activity")
            if not activity:
                raise StageError(
                    "Atividade SAIF não encontrada no manifesto de artefatos; "
                    "execute post-synth-sim ou use power.activity_source='vectorless'"
                )
            saif = Path(activity["path"])
            parse_saif(saif)
            strip_path = activity["strip_path"]
            unmatched = power_dir / "saif_unmatched.rpt"
            script = power_dir / "power_from_saif.tcl"

        atomic_write_text(
            script,
            power_tcl(self.config, dcp, saif, strip_path, report, unmatched),
        )
        self._run_tool(
            "power",
            [self.config.vivado, "-mode", "batch", "-source", str(script)],
            power_dir,
        )
        power = self._record_power_report(report, unmatched)
        # A política de cobertura mede a qualidade da anotação SAIF; sem SAIF
        # não há o que medir, e aplicá-la rejeitaria todo relatório vectorless.
        if (
            not vectorless
            and self.config.fail_on_low_confidence
            and not power["saif_coverage_passed"]
        ):
            raise StageError(
                "Cobertura SAIF insuficiente: {:.2f}% < {:.2f}%".format(
                    power["saif_match_percent"],
                    self.config.saif_min_match_percent,
                )
            )

    def _record_power_report(
        self,
        report: Path,
        unmatched: Optional[Path],
    ) -> Dict[str, Any]:
        """Persiste o diagnóstico antes de aplicar a política de cobertura."""
        report = Path(report).resolve()
        info = parse_power_report(report)
        vectorless = self.config.activity_source == "vectorless"
        # Sem SAIF a cobertura é inaplicável, e não "reprovada": marcá-la como
        # falha confundiria uma estimativa vectorless legítima com uma anotação
        # ruim. O que a distingue é activity_source.
        coverage_passed = (
            None
            if vectorless or info.saif_match_percent is None
            else info.saif_match_percent >= self.config.saif_min_match_percent
        )
        power = info.to_dict()
        activity = self.artifacts().get("activity")
        capture_is_partial = bool(
            not vectorless
            and isinstance(activity, dict)
            and activity.get("capture_is_partial")
        )
        power.update(
            {
                "report": str(report),
                "unmatched": str(Path(unmatched).resolve()) if unmatched else None,
                "activity_source": self.config.activity_source,
                "minimum_match_percent": (
                    None if vectorless else self.config.saif_min_match_percent
                ),
                "process": self.config.process,
                "ambient_temperature_c": self.config.ambient_temperature_c,
                "saif_coverage_passed": coverage_passed,
                "coverage_policy_enforced": (
                    False if vectorless else self.config.fail_on_low_confidence
                ),
                "quality_accepted": (
                    True
                    if vectorless
                    else (coverage_passed or not self.config.fail_on_low_confidence)
                ),
                # Uma estimativa vectorless usa taxas de transição padrão em vez
                # da atividade real do workload, então nunca é definitiva.
                "provisional": (
                    True if vectorless else ((not coverage_passed) or capture_is_partial)
                ),
                "capture_is_partial": capture_is_partial,
            }
        )
        if vectorless:
            power["default_toggle_rate_percent"] = self.config.default_toggle_rate_percent
            power["default_static_probability"] = self.config.default_static_probability
        atomic_write_json(self.reports_dir / "power_summary.json", power)
        self._set_artifact("power", power)
        return power

    def _analytic_duration_seconds(
        self, workload: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """Janela de execução sem simulação pós-síntese.

        Duas origens, nesta ordem de preferência:

        1. ``Total Execution Time`` da co-simulação, que é o testbench inteiro
           medido no RTL, com os stalls de contrapressão incluídos.  É a única
           fonte utilizável quando a síntese reporta latência ``undef``, o que
           acontece em regiões dataflow com FIFOs finitos.
        2. Latência do HLS x período x passos lógicos.  Só vale onde a latência
           é determinística; nesses casos reproduz a janela SAIF medida com erro
           de cerca de 0,1%.
        """
        cosim = self.artifacts().get("cosim") or {}
        total_cycles = cosim.get("total_execution_cycles")
        if total_cycles:
            try:
                seconds = float(total_cycles) * self.config.clock_period_ns * 1e-9
            except (TypeError, ValueError):
                seconds = 0.0
            if math.isfinite(seconds) and seconds > 0:
                return seconds

        latency = (self._hls_summary().get("latency") or {}).get("worst_case")
        try:
            cycles = float(latency)
        except (TypeError, ValueError):
            return None
        try:
            steps = int((workload or {}).get("total_logical_steps", 0))
        except (TypeError, ValueError):
            return None
        if cycles <= 0 or steps <= 0:
            return None
        seconds = cycles * self.config.clock_period_ns * 1e-9 * steps
        return seconds if math.isfinite(seconds) and seconds > 0 else None

    def _workload_summary(self) -> Optional[Dict[str, Any]]:
        candidates = (
            self.project_dir / "testbench.cpp",
            self.project_info.source_root / "testbench.cpp",
        )
        for path in candidates:
            workload = parse_testbench_workload(path)
            if workload is not None:
                return workload.to_dict()
        return None

    def _hls_summary(self) -> Dict[str, Any]:
        report = self.hls_solution_dir / "syn" / "report" / "{}_csynth.xml".format(self.project_info.top)
        if not report.is_file():
            return {"report": str(report), "available": False}
        try:
            import xml.etree.ElementTree as ET

            root = ET.parse(report).getroot()
            resources = root.find("./AreaEstimates/Resources")
            latency = root.find("./PerformanceEstimates/SummaryOfOverallLatency")
            result: Dict[str, Any] = {"report": str(report), "available": True}
            if resources is not None:
                result["resources"] = {
                    name: resources.findtext(name)
                    for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
                }
            if latency is not None:
                result["latency"] = {
                    "best_case": latency.findtext("Best-caseLatency"),
                    "average_case": latency.findtext("Average-caseLatency"),
                    "worst_case": latency.findtext("Worst-caseLatency"),
                }
            return result
        except Exception as error:
            return {"report": str(report), "available": True, "parse_error": str(error)}

    def _csim_summary(self) -> Dict[str, Any]:
        log = self.logs_dir / "csim.log"
        accuracy_percent = parse_csim_accuracy(log)
        return {
            "log": str(log),
            "available": accuracy_percent is not None,
            "accuracy_percent": accuracy_percent,
        }

    def _vivado_utilization_summary(self) -> Optional[Dict[str, Any]]:
        artifacts = self.artifacts()
        vivado = artifacts.get("vivado_synth")
        if isinstance(vivado, dict):
            utilization_summary = vivado.get("utilization_summary")
            if isinstance(utilization_summary, dict):
                return utilization_summary
        summary_path = self.reports_dir / "utilization_summary.json"
        if summary_path.is_file():
            try:
                payload = read_json(summary_path)
            except Exception:
                return None
            if isinstance(payload, dict):
                return payload
        return None

    def _write_summary(self) -> None:
        status = self.status()
        artifacts = self.artifacts()
        activity = artifacts.get("activity")
        workload = self._workload_summary()
        if workload is None and activity:
            workload = activity.get("workload")
        power = artifacts.get("power")
        vivado_utilization = self._vivado_utilization_summary()
        csim = self._csim_summary()
        cosim = artifacts.get("cosim") or {}
        derived_metrics = derive_summary_metrics(
            workload, activity, power,
            analytic_duration_seconds=self._analytic_duration_seconds(workload),
            analytic_duration_label=(
                "cosim_total_execution_time"
                if cosim.get("total_execution_cycles")
                else "hls_latency_times_logical_steps"
            ),
        )
        summary = {
            "run_id": status.get("run_id"),
            "state": status.get("state"),
            "project": self.project_info.to_dict(),
            "configuration": self.config.to_dict(),
            "execution_scope": status.get("execution_scope"),
            "stages": status.get("stages", {}),
            "csim": csim,
            "hls": self._hls_summary(),
            "vivado_utilization": vivado_utilization,
            "workload": workload,
            "cosim": cosim or None,
            "activity": activity,
            "power": power,
            "derived_metrics": derived_metrics,
            "artifacts": artifacts,
        }
        atomic_write_json(self.reports_dir / "summary.json", summary)
        lines = [
            "# Resumo da execução NeuroHLS",
            "",
            "- Run: `{}`".format(summary.get("run_id", "desconhecida")),
            "- Projeto: `{}`".format(self.project_info.source_root),
            "- Top: `{}`".format(self.project_info.top),
            "- FPGA: `{}`".format(self.config.part),
            "- Clock: `{:.6g} MHz` (`{:.6g} ns`)".format(
                self.config.frequency_mhz, self.config.clock_period_ns
            ),
        ]
        execution_scope = summary.get("execution_scope")
        if isinstance(execution_scope, dict):
            lines.extend(
                (
                    "",
                    "## Escopo da execução",
                    "",
                    "- Etapa inicial: `{}`".format(
                        execution_scope.get("from_stage", "N/D")
                    ),
                    "- Etapa final: `{}`".format(
                        execution_scope.get("to_stage", "N/D")
                    ),
                )
            )
        lines.extend(("", "## Etapas", ""))
        for name in STAGE_ORDER:
            entry = summary["stages"].get(name, {})
            missing_state = "não executada"
            if isinstance(execution_scope, dict):
                to_stage = execution_scope.get("to_stage")
                if (
                    to_stage in STAGE_ORDER
                    and STAGE_ORDER.index(name) > STAGE_ORDER.index(to_stage)
                ):
                    missing_state = "fora do escopo"
            lines.append(
                "- `{}`: {}".format(name, entry.get("state", missing_state))
            )
        accuracy_percent = csim.get("accuracy_percent")
        lines.extend(
            (
                "",
                "## Acurácia CSim",
                "",
                "- Acurácia final: `{}`".format(
                    "{}%".format(_format_pt_decimal(accuracy_percent, 2))
                    if accuracy_percent is not None
                    else "N/D"
                ),
                "- Log: `{}`".format(csim["log"]),
            )
        )
        if activity:
            activity_lines = [
                "",
                "## Atividade SAIF",
                "",
                "- Arquivo: `{}`".format(activity.get("path")),
                "- Duração: `{}`".format(activity.get("duration")),
                "- Transições: `{}`".format(activity.get("transition_count")),
                "- Strip path: `{}`".format(activity.get("strip_path")),
            ]
            if activity.get("capture_is_partial"):
                activity_lines.extend(
                    (
                        "- Captura parcial: limitada a `{}` transações por `SIM_POST_SYNTH_TRANSACTION_LIMIT`.".format(
                            activity.get("transaction_limit", "N/D")
                        ),
                        "- Os valores de potência/energia desta execução são estimativas provisórias da janela parcial e não substituem uma captura do workload completo.",
                    )
                )
            lines.extend(activity_lines)

        hls_summary = summary.get("hls") or {}
        hls_resources = hls_summary.get("resources") if isinstance(hls_summary, dict) else None
        if isinstance(hls_summary, dict) and hls_summary.get("available") and hls_resources:
            lines.extend(
                (
                    "",
                    "## Estimativa de recursos HLS",
                    "",
                    "Valores reportados pelo csynth.xml antes da síntese Vivado; não representam o uso pós-síntese.",
                    "",
                    "| Recurso | Estimado |",
                    "|---|---:|",
                )
            )
            for label in ("LUT", "FF", "BRAM_18K", "DSP", "URAM"):
                if label in hls_resources and hls_resources[label] is not None:
                    lines.append(
                        "| {} | `{}` |".format(label, hls_resources[label])
                    )
            latency = hls_summary.get("latency")
            if isinstance(latency, dict):
                lines.extend(
                    (
                        "",
                        "- Latência HLS: melhor caso `{}`, média `{}` e pior caso `{}`.".format(
                            latency.get("best_case", "n/d"),
                            latency.get("average_case", "n/d"),
                            latency.get("worst_case", "n/d"),
                        ),
                    )
                )

        if vivado_utilization and isinstance(vivado_utilization, dict):
            resources = vivado_utilization.get("resources", {})
            if isinstance(resources, dict) and resources:
                lines.extend(
                    (
                        "",
                        "## Uso de recursos pós-síntese — Vivado OOC",
                        "",
                        "Os valores abaixo são pós-síntese e foram gerados em modo out-of-context. O percentual usa a capacidade total do FPGA reportada pelo Vivado; recursos reservados por plataforma ou shell não são descontados. BRAM pode aparecer fracionário porque uma RAMB18 ocupa meio tile.",
                        "",
                        "| Recurso | Utilizado | Disponível | Uso |",
                        "|---|---:|---:|---:|",
                    )
                )
                table_order = (
                    ("lut", "LUT", 0),
                    ("ff", "FF", 0),
                    ("bram", "BRAM", 1),
                    ("dsp", "DSP", 0),
                    ("uram", "URAM", 0),
                )
                for resource_name, label, fractional_digits in table_order:
                    resource = resources.get(resource_name)
                    if not isinstance(resource, dict):
                        continue
                    lines.append(
                        "| {} | `{}` | `{}` | `{}%` |".format(
                            label,
                            _format_resource_quantity(resource.get("used"), fractional_digits),
                            _format_resource_quantity(resource.get("available"), fractional_digits),
                            _format_resource_percent(resource.get("recalculated_utilization_percent")),
                        )
                    )
                lines.extend(
                    (
                        "",
                        "Os valores estruturados preservam o percentual informado pelo Vivado e o percentual recalculado a partir de `Used / Available` para validação automática.",
                    )
                )
            else:
                lines.extend(
                    (
                        "",
                        "## Uso de recursos pós-síntese — Vivado OOC",
                        "",
                        "Relatório pós-síntese indisponível ou inválido; a utilização pós-síntese não está disponível nesta execução.",
                    )
                )
        else:
            lines.extend(
                (
                    "",
                    "## Uso de recursos pós-síntese — Vivado OOC",
                    "",
                    "Relatório pós-síntese indisponível ou inválido; a utilização pós-síntese não está disponível nesta execução.",
                )
            )

        if workload:
            executed_samples = workload["executed_samples"]
            steps_per_sample = workload["steps_per_sample"]
            total_steps = workload["total_logical_steps"]
            lines.extend(
                (
                    "",
                    "## Carga da simulação",
                    "",
                    "- Amostras: `{}`".format(
                        _format_pt_integer(executed_samples)
                    ),
                    "- Passos temporais por amostra: `{}`".format(
                        _format_pt_integer(steps_per_sample)
                    ),
                    "- Passos temporais executados: `{}` (`{} × {}`)".format(
                        _format_pt_integer(total_steps),
                        _format_pt_integer(executed_samples),
                        _format_pt_integer(steps_per_sample),
                    ),
                    "- Batches: `{}`".format(
                        _format_pt_integer(workload["batch_count"])
                    ),
                    "- Tamanho do batch: `{}`".format(
                        _format_pt_integer(workload["batch_size"])
                    ),
                    "",
                    "Os passos acima são passos temporais lógicos do workload. O batch",
                    "organiza a execução do testbench e não é tratado como paralelismo do",
                    "hardware; em backends event-driven, o número de transações do DUT pode",
                    "ser diferente do número de passos lógicos.",
                )
            )
            if workload.get("ignored_samples"):
                lines.extend(
                    (
                        "",
                        "> Aviso: `{}` amostra(s) declarada(s) não foram executadas, pois"
                        " `TOTAL_SAMPLES` não é múltiplo de `BATCH_SIZE`.".format(
                            _format_pt_integer(workload["ignored_samples"])
                        ),
                    )
                )

        duration_seconds = derived_metrics.get("capture_duration_seconds")
        step_latency = derived_metrics.get("average_latency_per_step_seconds")
        sample_latency = derived_metrics.get("average_latency_per_sample_seconds")
        if duration_seconds is not None:
            lines.extend(
                (
                    "",
                    "## Duração e latência",
                    "",
                    "| Métrica | Valor | Cálculo |",
                    "|---|---:|---:|",
                )
            )
            duration_ms = float(duration_seconds) * 1e3
            lines.append(
                "| Duração simulada total (tempo lógico) | `{} ms` | `{}` |".format(
                    _format_pt_decimal(duration_ms, 9),
                    _format_saif_duration_expression(activity or {}),
                )
            )
            if step_latency is not None and workload:
                lines.append(
                    "| Latência média amortizada por passo temporal | `{} µs/step`"
                    " | `{} ms / {}` |".format(
                        _format_pt_decimal(float(step_latency) * 1e6, 6),
                        _format_pt_decimal(duration_ms, 9),
                        _format_pt_integer(workload["total_logical_steps"]),
                    )
                )
            if sample_latency is not None and workload:
                lines.append(
                    "| Latência média amortizada por amostra | `{} µs/amostra`"
                    " | `{} ms / {}` |".format(
                        _format_pt_decimal(float(sample_latency) * 1e6, 6),
                        _format_pt_decimal(duration_ms, 9),
                        _format_pt_integer(workload["executed_samples"]),
                    )
                )
            lines.extend(
                (
                    "",
                    "As latências são médias calculadas sobre a duração total registrada",
                    "no SAIF. Elas incluem inicialização, intervalos entre chamadas e",
                    "finalização do testbench; não são uma medição isolada de `ap_start`",
                    "até `ap_done`. O tempo de wall-clock do simulador não entra nesses",
                    "cálculos.",
                )
            )

        required_power_fields = (
            "total_on_chip_power_w",
            "dynamic_power_w",
            "device_static_power_w",
        )
        power_values_available = power and all(
            field in power for field in required_power_fields
        )
        if power_values_available:
            lines.extend(
                (
                    "",
                    "## Potência e energia",
                    "",
                    "| Métrica | Valor |",
                    "|---|---:|",
                    "| Potência total | `{} W` |".format(
                        _format_power_display(power, "total_on_chip_power")
                    ),
                    "| Potência dinâmica | `{} W` |".format(
                        _format_power_display(power, "dynamic_power")
                    ),
                    "| Potência estática | `{} W` |".format(
                        _format_power_display(power, "device_static_power")
                    ),
                )
            )
            capture_energy = derived_metrics.get("capture_energy_total_joules")
            step_energy = derived_metrics.get("energy_per_step_total_joules")
            sample_energy = derived_metrics.get("energy_per_sample_total_joules")
            if capture_energy is not None:
                lines.append(
                    "| Energia total da simulação | `{} mJ` |".format(
                        _format_pt_decimal(float(capture_energy) * 1e3, 6)
                    )
                )
            if step_energy is not None:
                lines.append(
                    "| Energia média por passo temporal | `{} µJ/step` |".format(
                        _format_pt_decimal(float(step_energy) * 1e6, 6)
                    )
                )
            if sample_energy is not None:
                lines.append(
                    "| Energia média por amostra | `{} mJ/amostra` |".format(
                        _format_pt_decimal(float(sample_energy) * 1e3, 6)
                    )
                )
            if (
                capture_energy is not None
                and step_energy is not None
                and sample_energy is not None
                and workload
                and duration_seconds is not None
            ):
                lines.extend(
                    (
                        "",
                        "Os valores de energia usam a potência total:",
                        "",
                        "```text",
                        "E_total   = {} W × {} ms = {} mJ".format(
                            _format_power_display(power, "total_on_chip_power"),
                            _format_pt_decimal(float(duration_seconds) * 1e3, 9),
                            _format_pt_decimal(float(capture_energy) * 1e3, 9),
                        ),
                        "E_step    = E_total / {} = {} µJ".format(
                            _format_pt_integer(workload["total_logical_steps"]),
                            _format_pt_decimal(float(step_energy) * 1e6, 9),
                        ),
                        "E_amostra = E_total / {} = {} mJ".format(
                            _format_pt_integer(workload["executed_samples"]),
                            _format_pt_decimal(float(sample_energy) * 1e3, 9),
                        ),
                        "```",
                    )
                )

            confidence = power.get("confidence_level", "não informada")
            if power.get("activity_source") == "vectorless":
                lines.extend(
                    (
                        "",
                        "A potência foi estimada sem SAIF, com taxa de transição padrão de "
                        "{}% e probabilidade estática {}.".format(
                            _format_pt_decimal(
                                power.get("default_toggle_rate_percent", 0.0), 2
                            ),
                            _format_pt_decimal(
                                power.get("default_static_probability", 0.0), 2
                            ),
                        ),
                        "O Vivado informou confiança geral `{}`. Estes valores não".format(
                            confidence
                        ),
                        "refletem a atividade real do workload e são estimativas provisórias.",
                    )
                )

            match_percent = power.get("saif_match_percent")
            minimum_match = power.get("minimum_match_percent")
            if match_percent is not None and minimum_match is not None:
                match_digits = (
                    0 if float(match_percent).is_integer() else 2
                )
                minimum_digits = (
                    0 if float(minimum_match).is_integer() else 2
                )
                counts = ""
                matched_nets = power.get("saif_matched_design_nets")
                total_nets = power.get("saif_total_design_nets")
                if matched_nets is not None and total_nets is not None:
                    counts = " (`{}/{}`)".format(
                        _format_pt_integer(matched_nets),
                        _format_pt_integer(total_nets),
                    )
                lines.extend(
                    (
                        "",
                        "O Vivado informou confiança geral `{}`, com {}% dos nets".format(
                            confidence,
                            _format_pt_decimal(match_percent, match_digits),
                        ),
                        "anotados pelo SAIF{}, para um limite de {}%.".format(
                            counts,
                            _format_pt_decimal(minimum_match, minimum_digits),
                        ),
                    )
                )
                if not power.get("saif_coverage_passed", False):
                    power_state = summary["stages"].get("power", {}).get("state")
                    if power_state == "failed":
                        lines.extend(
                            (
                                "Por isso, a etapa `power` permanece marcada como `failed` e",
                                "estes valores de potência e energia são estimativas provisórias.",
                            )
                        )
                    else:
                        lines.extend(
                            (
                                "A cobertura está abaixo do limite; estes valores de potência",
                                "e energia são estimativas provisórias, mesmo que a política de",
                                "falha tenha sido desativada explicitamente.",
                            )
                        )
        elif power:
            lines.extend(
                (
                    "",
                    "## Potência",
                    "",
                    "- Relatório: `{}`".format(power.get("report")),
                    "- Nets SAIF correspondentes: `{}`".format(
                        power.get("saif_match_percent", "não disponível")
                    ),
                    "- Valores de potência: `não disponíveis no artefato legado`",
                )
            )
        atomic_write_text(self.reports_dir / "summary.md", "\n".join(lines) + "\n")

    def preflight(self) -> Dict[str, Any]:
        """Executa validações leves sem iniciar Vitis/Vivado."""
        checks: List[Dict[str, Any]] = []
        required_tools = [
            ("vitis_run", self.config.vitis_run),
            ("vivado", self.config.vivado),
        ]
        # xelab/xsim só participam das etapas que produzem o SAIF; exigi-los no
        # modo vectorless reprovaria uma máquina que roda o fluxo inteiro.
        if self.config.activity_source != "vectorless":
            required_tools.extend(
                (("xelab", self.config.xelab), ("xsim", self.config.xsim))
            )
        for label, executable in required_tools:
            resolved = shutil.which(executable)
            if self.config.settings_script:
                passed = Path(self.config.settings_script).expanduser().is_file()
                detail = "será resolvido após source de settings_script"
            else:
                passed = resolved is not None
                detail = resolved or "não encontrado no PATH"
            checks.append({"name": label, "passed": passed, "detail": detail})
        try:
            current_info = validate_project(self.project_info.source_root)
            hash_matches = current_info.project_hash == self.project_info.project_hash
            checks.append(
                {
                    "name": "project_contract",
                    "passed": hash_matches,
                    "detail": "estrutura válida" if hash_matches else "projeto de origem mudou após a criação da run",
                }
            )
        except ValidationError as error:
            checks.append({"name": "project_contract", "passed": False, "detail": str(error)})
        payload = {
            "created_at": utc_now(),
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
            "configuration": self.config.to_dict(),
        }
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.reports_dir / "preflight.json", payload)
        if not payload["passed"]:
            failures = [check["name"] for check in checks if not check["passed"]]
            raise ValidationError("Preflight falhou: {}".format(", ".join(failures)))
        return payload
