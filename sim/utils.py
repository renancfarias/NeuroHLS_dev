"""Funções utilitárias sem dependências das ferramentas AMD."""

import hashlib
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .errors import CommandError, ValidationError


def utc_now() -> str:
    """Retorna uma data UTC estável para manifestos e identificadores."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    result = result.strip(".-")
    if not result:
        raise ValidationError("Identificador vazio ou sem caracteres seguros")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: Dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def tcl_quote(value: object) -> str:
    """Cota um valor como uma palavra Tcl sem permitir substituição."""
    # Dentro de chaves Tcl não expande $, [] ou barras invertidas. Preservar a
    # barra é importante para caminhos Windows; apenas chaves precisam escape.
    text = str(value).replace("{", "\\{").replace("}", "\\}")
    return "{" + text + "}"


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    log_path: Path
    elapsed_seconds: float


def _process_group_exists(process_group_id: int) -> bool:
    """Consulta um grupo POSIX tolerando a corrida com o término do processo."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen,
    grace_seconds: float = 5.0,
) -> None:
    """Encerra a ferramenta e seus filhos antes de devolver o controle."""
    if os.name == "nt":
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            except (AttributeError, OSError, ValueError):
                process.terminate()
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass
        return

    deadline = time.monotonic() + grace_seconds
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass

    while time.monotonic() < deadline and _process_group_exists(process_group_id):
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class CommandRunner:
    """Executa ferramentas externas e preserva stdout/stderr em log."""

    def __init__(self, timeout_seconds: Optional[int] = None, dry_run: bool = False):
        self.timeout_seconds = timeout_seconds
        self.dry_run = dry_run

    def run(
        self,
        command: Sequence[str],
        cwd: Path,
        log_path: Path,
        environment: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        command = [str(item) for item in command]
        cwd = Path(cwd)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()

        header = "$ {}\n# cwd: {}\n".format(command_text(command), cwd)
        if self.dry_run:
            atomic_write_text(log_path, header + "# dry-run: comando não executado\n")
            return CommandResult(command, 0, log_path, 0.0)

        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            log_handle.write(header)
            log_handle.flush()
            try:
                process_options = (
                    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    if os.name == "nt"
                    else {"start_new_session": True}
                )
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    errors="replace",
                    **process_options,
                )
            except OSError as error:
                log_handle.write("Não foi possível iniciar o comando: {}\n".format(error))
                raise CommandError(command, 127, log_path) from error

            assert process.stdout is not None
            output_queue = queue.Queue()

            def pump_output() -> None:
                try:
                    for line in process.stdout:
                        output_queue.put(line)
                finally:
                    output_queue.put(None)

            thread = threading.Thread(target=pump_output, daemon=True)
            thread.start()
            try:
                while True:
                    try:
                        line = output_queue.get(timeout=0.2)
                    except queue.Empty:
                        if (
                            self.timeout_seconds is not None
                            and time.monotonic() - started > self.timeout_seconds
                        ):
                            raise subprocess.TimeoutExpired(command, self.timeout_seconds)
                        # Alguns front-ends deixam filhos com o pipe aberto após
                        # terminar. Não espere indefinidamente por EOF nesse caso.
                        if process.poll() is not None:
                            break
                        continue
                    if line is None:
                        break
                    print(line, end="")
                    log_handle.write(line)
                    log_handle.flush()
                returncode = process.wait(timeout=5)
            except KeyboardInterrupt:
                _terminate_process_group(process)
                thread.join(timeout=1)
                log_handle.write("\nProcesso interrompido pelo usuário.\n")
                log_handle.flush()
                raise
            except subprocess.TimeoutExpired as error:
                _terminate_process_group(process)
                thread.join(timeout=1)
                log_handle.write("\nProcesso excedeu o timeout configurado.\n")
                log_handle.flush()
                raise CommandError(command, 124, log_path) from error

        elapsed = time.monotonic() - started
        if returncode != 0:
            raise CommandError(command, returncode, log_path)
        return CommandResult(command, returncode, log_path, elapsed)
