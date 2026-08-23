#!/usr/bin/env python3
"""Compatibilidade: encaminha para o gerador completo de resultados."""

import sys

sys.dont_write_bytecode = True

from gerar_resultados import main


if __name__ == "__main__":
    main()
