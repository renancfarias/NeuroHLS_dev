"""Validação e cópia de projetos clássicos gerados pelo NeuroHLS."""

import hashlib
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .errors import ValidationError
from .utils import sha256_file, slug


REQUIRED_FILES = (
    "0_create_project.tcl",
    "1_csim.tcl",
    "2_synth.tcl",
    "3_cosim.tcl",
    "snn_implementation.cpp",
    "snn_implementation.h",
    "testbench.cpp",
    "neuron_params.h",
    "quantization.h",
    "tb_data/data.txt",
    "tb_data/targets.txt",
)

MEDOID_TESTBENCH_FILES = (
    "tb_medoids/testbench.cpp",
    "tb_medoids/data.txt",
    "tb_medoids/targets.txt",
)

VOLATILE_DIRECTORIES = {
    ".cache",
    ".Xil",
    "_ide",
    "__pycache__",
    "logs",
    "vitis_proj",
}
VOLATILE_SUFFIXES = {
    ".dcp",
    ".jou",
    ".log",
    ".saif",
    ".wdb",
    ".xpr",
    ".zip",
}


@dataclass
class ProjectInfo:
    source_root: Path
    identifier: str
    top: str
    project_hash: str
    required_files: List[str]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["source_root"] = str(self.source_root)
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectInfo":
        return cls(
            source_root=Path(data["source_root"]),
            identifier=data["identifier"],
            top=data["top"],
            project_hash=data["project_hash"],
            required_files=list(data.get("required_files", REQUIRED_FILES)),
        )


def _is_volatile(relative_path: Path) -> bool:
    if any(part in VOLATILE_DIRECTORIES for part in relative_path.parts):
        return True
    return relative_path.suffix.lower() in VOLATILE_SUFFIXES


def _project_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_volatile(relative):
            continue
        yield path


def hash_project(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _project_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def discover_top(root: Path) -> str:
    synth_tcl = root / "2_synth.tcl"
    contents = synth_tcl.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*set_top\s+\{?([A-Za-z_][A-Za-z0-9_]*)\}?", contents, re.MULTILINE)
    if match:
        return match.group(1)

    header = root / "snn_implementation.h"
    header_contents = header.read_text(encoding="utf-8", errors="replace")
    candidates = re.findall(
        r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", header_contents
    )
    if len(candidates) == 1:
        return candidates[0]
    raise ValidationError(
        "Não foi possível descobrir o top HLS. Esperado 'set_top <nome>' em {}"
        .format(synth_tcl)
    )


def validate_project(project_path: Path, runs_root: Optional[Path] = None) -> ProjectInfo:
    project_path = Path(project_path).expanduser()
    if project_path.suffix.lower() == ".nir":
        raise ValidationError("O ambiente aceita uma pasta de projeto NeuroHLS, não um arquivo .nir")
    if not project_path.exists():
        raise ValidationError("Projeto inexistente: {}".format(project_path))
    if not project_path.is_dir():
        raise ValidationError("--project deve apontar para uma pasta NeuroHLS: {}".format(project_path))
    root = project_path.resolve()
    if runs_root is not None:
        try:
            root.relative_to(Path(runs_root).resolve())
        except ValueError:
            pass
        else:
            raise ValidationError("A pasta de entrada não pode estar dentro de sim/runs")

    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    support = root / "neuro_hls_functions"
    if not support.is_dir():
        missing.append("neuro_hls_functions/")
    if missing:
        raise ValidationError(
            "Pasta não atende ao contrato de projeto NeuroHLS; faltam: {}".format(
                ", ".join(missing)
            )
        )
    empty_data = [
        relative
        for relative in ("tb_data/data.txt", "tb_data/targets.txt")
        if (root / relative).stat().st_size == 0
    ]
    if empty_data:
        raise ValidationError("Dados de testbench vazios: {}".format(", ".join(empty_data)))

    medoid_files = [root / relative for relative in MEDOID_TESTBENCH_FILES]
    medoid_present = [path.is_file() for path in medoid_files]
    if any(medoid_present) and not all(medoid_present):
        missing_medoids = [
            relative
            for relative, present in zip(MEDOID_TESTBENCH_FILES, medoid_present)
            if not present
        ]
        raise ValidationError(
            "Bundle de medoids incompleto; faltam: {}".format(
                ", ".join(missing_medoids)
            )
        )
    if all(medoid_present):
        empty_medoids = [
            relative
            for relative, path in zip(MEDOID_TESTBENCH_FILES, medoid_files)
            if path.stat().st_size == 0
        ]
        if empty_medoids:
            raise ValidationError(
                "Arquivos de medoids vazios: {}".format(", ".join(empty_medoids))
            )

    return ProjectInfo(
        source_root=root,
        identifier=slug(root.name),
        top=discover_top(root),
        project_hash=hash_project(root),
        required_files=list(REQUIRED_FILES),
    )


def copy_project(source: Path, destination: Path) -> None:
    """Copia somente a entrada funcional, nunca resultados de ferramentas."""
    source = Path(source).resolve()
    destination = Path(destination)
    if destination.exists():
        raise ValidationError("Destino da cópia já existe: {}".format(destination))

    def ignore(directory: str, names: List[str]) -> List[str]:
        ignored = []
        root = Path(directory)
        for name in names:
            candidate = root / name
            try:
                relative = candidate.relative_to(source)
            except ValueError:
                continue
            if candidate.is_dir() and name in VOLATILE_DIRECTORIES:
                ignored.append(name)
            elif candidate.is_file() and _is_volatile(relative):
                ignored.append(name)
        return ignored

    shutil.copytree(source, destination, ignore=ignore, copy_function=shutil.copy2)
