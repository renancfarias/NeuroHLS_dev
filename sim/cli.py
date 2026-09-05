"""Interface de linha de comando para ``python -m sim``."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_config
from .errors import SimEnvironmentError
from .pipeline import STAGE_ORDER, Pipeline
from .project import validate_project


SIM_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SIM_ROOT / "runs"


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True, type=Path, help="Pasta gerada pelo NeuroHLS")
    parser.add_argument("--environment", type=Path, help="Arquivo YAML de configuração da plataforma")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Raiz das execuções")
    parser.add_argument("--part", help="Substitui target.part")
    parser.add_argument("--frequency-mhz", type=float, help="Substitui a frequência de clock")
    parser.add_argument("--clock-name", help="Substitui o nome da porta de clock")
    parser.add_argument("--timeout-minutes", type=int, help="Substitui o timeout de cada ferramenta")
    parser.add_argument("--saif-min-match-percent", type=float, help="Limite mínimo de nets SAIF anotadas")
    parser.add_argument("--settings-script", help="Script de ambiente AMD a ser sourced antes de cada ferramenta")
    parser.add_argument("--dry-run", action="store_true", help="Gera a run e scripts sem executar ferramentas")


def _overrides_from_arguments(arguments: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if arguments.part:
        overrides.setdefault("target", {})["part"] = arguments.part
    if arguments.frequency_mhz is not None:
        overrides.setdefault("target", {}).setdefault("clock", {})["frequency_mhz"] = arguments.frequency_mhz
    if arguments.clock_name:
        overrides.setdefault("target", {}).setdefault("clock", {})["name"] = arguments.clock_name
    if arguments.timeout_minutes is not None:
        overrides.setdefault("execution", {})["timeout_minutes"] = arguments.timeout_minutes
    if arguments.saif_min_match_percent is not None:
        overrides.setdefault("power", {})["saif_min_match_percent"] = arguments.saif_min_match_percent
    if arguments.settings_script:
        overrides.setdefault("tools", {})["settings_script"] = arguments.settings_script
    return overrides


def _create_pipeline(arguments: argparse.Namespace) -> Pipeline:
    config = load_config(arguments.environment, _overrides_from_arguments(arguments))
    return Pipeline.create(
        arguments.project,
        config,
        arguments.runs_dir,
        dry_run=arguments.dry_run,
    )


def _print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim",
        description="Fluxo isolado de Vitis HLS, Vivado, SAIF e potência para projetos NeuroHLS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Valida o contrato de uma pasta NeuroHLS")
    validate.add_argument("--project", required=True, type=Path)
    validate.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    preflight = subparsers.add_parser("preflight", help="Cria uma run e verifica ferramentas/entrada")
    _add_common_run_arguments(preflight)

    run = subparsers.add_parser("run", help="Executa o fluxo completo")
    _add_common_run_arguments(run)
    run.add_argument("--from", dest="from_stage", choices=STAGE_ORDER)
    run.add_argument("--to", dest="to_stage", choices=STAGE_ORDER)
    run.add_argument("--force", action="store_true", help="Não reutiliza etapas concluídas")
    run.add_argument("--no-reuse", action="store_true", help="Desativa reutilização de etapas válidas")

    resume = subparsers.add_parser("resume", help="Retoma uma run existente")
    resume.add_argument("--run", required=True, type=Path)
    resume.add_argument("--from", dest="from_stage", choices=STAGE_ORDER)
    resume.add_argument("--to", dest="to_stage", choices=STAGE_ORDER)
    resume.add_argument("--force", action="store_true")
    resume.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status", help="Exibe o estado de uma run")
    status.add_argument("--run", required=True, type=Path)
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            info = validate_project(arguments.project, runs_root=arguments.runs_dir)
            _print_json(info.to_dict())
            return 0

        if arguments.command == "status":
            pipeline = Pipeline.load(arguments.run)
            _print_json({"status": pipeline.status(), "artifacts": pipeline.artifacts()})
            return 0

        if arguments.command == "preflight":
            pipeline = _create_pipeline(arguments)
            result = pipeline.preflight()
            _print_json({"run_dir": str(pipeline.run_dir), "preflight": result})
            return 0

        if arguments.command == "run":
            pipeline = _create_pipeline(arguments)
            if arguments.no_reuse:
                pipeline.config.reuse_cache = False
            pipeline.preflight()
            pipeline.run(arguments.from_stage, arguments.to_stage, force=arguments.force)
            _print_json({"run_dir": str(pipeline.run_dir), "status": pipeline.status()})
            return 0

        if arguments.command == "resume":
            pipeline = Pipeline.load(arguments.run, dry_run=arguments.dry_run)
            pipeline.preflight()
            pipeline.run(arguments.from_stage, arguments.to_stage, force=arguments.force)
            _print_json({"run_dir": str(pipeline.run_dir), "status": pipeline.status()})
            return 0
    except SimEnvironmentError as error:
        print("ERRO: {}".format(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.", file=sys.stderr)
        return 130
    return 1
