"""Leitura, validação e resolução da configuração do ambiente."""

import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import ValidationError


DEFAULT_CONFIG = {
    "tools": {
        "vitis_run": "vitis-run",
        "vivado": "vivado",
        "xelab": "xelab",
        "xsim": "xsim",
        "settings_script": None,
    },
    "target": {
        "part": "xcu250-figd2104-2L-e",
        "clock": {"name": "ap_clk", "frequency_mhz": 150.0, "uncertainty_ns": 0.2},
    },
    "execution": {
        "jobs": 1,
        "timeout_minutes": 180,
        "reuse_cache": True,
        "keep_intermediates": True,
    },
    "simulation": {
        "profile": "post-synth",
        "verify_outputs": True,
        "activity": {
            "warmup_samples": 0,
            "measured_samples": "all_remaining",
            "capture_scope": "dut",
        },
        # Mede a latência real no RTL. Necessária quando a síntese reporta
        # latência 'undef', o que acontece em regiões dataflow com FIFOs
        # finitos: o tempo depende da contrapressão entre atores.
        "run_cosim": False,
    },
    "power": {
        "process": "typical",
        "ambient_temperature_c": 25.0,
        # "vectorless" estima a potência a partir das taxas de transição padrão
        # do Vivado; "saif" anota a atividade medida na simulação pós-síntese.
        # O padrão é vectorless porque a simulação de gate que produz o SAIF
        # custa horas por design, enquanto o relatório vectorless custa minutos.
        "activity_source": "vectorless",
        "default_toggle_rate_percent": 12.5,
        "default_static_probability": 0.5,
        "saif_min_match_percent": 50.0,
        "fail_on_low_confidence": True,
    },
    "hls": {"project_name": "vitis_proj", "solution_name": "sol"},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ValidationError(
            "PyYAML é necessário para ler --environment. Instale pyyaml ou omita essa opção."
        ) from error
    try:
        with path.open("r", encoding="utf-8") as handle:
            contents = yaml.safe_load(handle) or {}
    except OSError as error:
        raise ValidationError("Não foi possível ler a configuração {}: {}".format(path, error)) from error
    if not isinstance(contents, dict):
        raise ValidationError("A configuração {} deve conter um mapa YAML".format(path))
    return contents


@dataclass
class RunConfig:
    vitis_run: str = "vitis-run"
    vivado: str = "vivado"
    xelab: str = "xelab"
    xsim: str = "xsim"
    settings_script: Optional[str] = None
    part: str = "xcu250-figd2104-2L-e"
    clock_name: str = "ap_clk"
    frequency_mhz: float = 150.0
    clock_uncertainty_ns: float = 0.2
    jobs: int = 1
    timeout_minutes: int = 180
    reuse_cache: bool = True
    keep_intermediates: bool = True
    profile: str = "post-synth"
    verify_outputs: bool = True
    run_cosim: bool = False
    warmup_samples: int = 0
    measured_samples: str = "all_remaining"
    capture_scope: str = "dut"
    process: str = "typical"
    ambient_temperature_c: float = 25.0
    activity_source: str = "vectorless"
    default_toggle_rate_percent: float = 12.5
    default_static_probability: float = 0.5
    saif_min_match_percent: float = 50.0
    fail_on_low_confidence: bool = True
    hls_project_name: str = "vitis_proj"
    hls_solution_name: str = "sol"

    @property
    def clock_period_ns(self) -> float:
        return 1000.0 / self.frequency_mhz

    @property
    def timeout_seconds(self) -> int:
        return self.timeout_minutes * 60

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["clock_period_ns"] = self.clock_period_ns
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def validate(self) -> "RunConfig":
        if not self.part or any(character.isspace() for character in self.part):
            raise ValidationError("target.part deve ser um part FPGA não vazio e sem espaços")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", self.clock_name):
            raise ValidationError(
                "target.clock.name deve ser um identificador HDL simples e não vazio"
            )
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ValidationError("target.clock.frequency_mhz deve ser maior que zero")
        if (
            not math.isfinite(self.clock_uncertainty_ns)
            or self.clock_uncertainty_ns < 0
        ):
            raise ValidationError(
                "target.clock.uncertainty_ns deve ser finita e não negativa"
            )
        if self.jobs <= 0:
            raise ValidationError("execution.jobs deve ser maior que zero")
        if self.timeout_minutes <= 0:
            raise ValidationError("execution.timeout_minutes deve ser maior que zero")
        if not 0.0 <= self.saif_min_match_percent <= 100.0:
            raise ValidationError("power.saif_min_match_percent deve estar entre 0 e 100")
        if self.activity_source not in ("vectorless", "saif"):
            raise ValidationError(
                "power.activity_source deve ser 'vectorless' ou 'saif', não {!r}".format(
                    self.activity_source
                )
            )
        if not 0.0 <= self.default_toggle_rate_percent <= 100.0:
            raise ValidationError(
                "power.default_toggle_rate_percent deve estar entre 0 e 100"
            )
        if not 0.0 <= self.default_static_probability <= 1.0:
            raise ValidationError(
                "power.default_static_probability deve estar entre 0 e 1"
            )
        if self.profile != "post-synth":
            raise ValidationError(
                "A implementação atual suporta somente simulation.profile='post-synth'; "
                "power-accurate (pós-route/SDF) permanece planejado."
            )
        if self.capture_scope != "dut":
            raise ValidationError(
                "simulation.activity.capture_scope suporta somente o valor 'dut'"
            )
        if not self.hls_project_name or not self.hls_solution_name:
            raise ValidationError("Os nomes do projeto e solução HLS não podem ser vazios")
        return self


def load_config(path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
    data = DEFAULT_CONFIG
    if path is not None:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValidationError("Arquivo de ambiente inexistente: {}".format(path))
        data = _deep_merge(data, _load_yaml(path))
    if overrides:
        data = _deep_merge(data, overrides)

    tools = data["tools"]
    target = data["target"]
    clock = target["clock"]
    execution = data["execution"]
    simulation = data["simulation"]
    activity = simulation.get("activity", {})
    power = data["power"]
    hls = data.get("hls", {})
    config = RunConfig(
        vitis_run=str(tools["vitis_run"]),
        vivado=str(tools["vivado"]),
        xelab=str(tools["xelab"]),
        xsim=str(tools["xsim"]),
        settings_script=tools.get("settings_script"),
        part=str(target["part"]),
        clock_name=str(clock.get("name", "ap_clk")),
        frequency_mhz=float(clock["frequency_mhz"]),
        clock_uncertainty_ns=float(clock.get("uncertainty_ns", 0.0)),
        jobs=int(execution["jobs"]),
        timeout_minutes=int(execution["timeout_minutes"]),
        reuse_cache=bool(execution.get("reuse_cache", True)),
        keep_intermediates=bool(execution.get("keep_intermediates", True)),
        profile=str(simulation["profile"]),
        verify_outputs=bool(simulation.get("verify_outputs", True)),
        run_cosim=bool(simulation.get("run_cosim", False)),
        warmup_samples=int(activity.get("warmup_samples", 0)),
        measured_samples=str(activity.get("measured_samples", "all_remaining")),
        capture_scope=str(activity.get("capture_scope", "dut")),
        process=str(power.get("process", "typical")),
        ambient_temperature_c=float(power.get("ambient_temperature_c", 25.0)),
        activity_source=str(power.get("activity_source", "vectorless")),
        default_toggle_rate_percent=float(power.get("default_toggle_rate_percent", 12.5)),
        default_static_probability=float(power.get("default_static_probability", 0.5)),
        saif_min_match_percent=float(power["saif_min_match_percent"]),
        fail_on_low_confidence=bool(power.get("fail_on_low_confidence", True)),
        hls_project_name=str(hls.get("project_name", "vitis_proj")),
        hls_solution_name=str(hls.get("solution_name", "sol")),
    )
    return config.validate()
