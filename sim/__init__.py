"""Ambiente reproduzível de simulação e potência para projetos NeuroHLS.

O pacote é executável diretamente a partir da raiz do repositório:

    python -m sim run --project <pasta-gerada-pelo-neurohls>
"""

from .config import RunConfig
from .pipeline import Pipeline
from .project import ProjectInfo, validate_project

__all__ = ["Pipeline", "ProjectInfo", "RunConfig", "validate_project"]

