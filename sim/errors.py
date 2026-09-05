"""Exceções com contexto para o ambiente de simulação."""


class SimEnvironmentError(RuntimeError):
    """Erro base do ambiente de simulação."""


class ValidationError(SimEnvironmentError):
    """Entrada ou configuração inválida."""


class StageError(SimEnvironmentError):
    """Uma etapa externa falhou ou não produziu seus artefatos."""


class CommandError(StageError):
    """Um processo externo terminou com erro."""

    def __init__(self, command, returncode, log_path):
        self.command = command
        self.returncode = returncode
        self.log_path = log_path
        super().__init__(
            "Comando externo falhou (código {}): {}\nLog: {}".format(
                returncode, " ".join(command), log_path
            )
        )

