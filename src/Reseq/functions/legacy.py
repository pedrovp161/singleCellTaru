"""
Acesso ao código da auditoria anterior (`src/scripts/claude/functions`).

As duas árvores têm um pacote chamado ``functions``. Para reutilizar módulos de
lá sem colisão de nome — e sem executar o ``__init__`` de lá, que importa o
pacote inteiro —, o diretório antigo é montado sob o alias ``claude_functions``.

    doublets_legacy = legacy.module("doublets")
"""

from __future__ import annotations

import importlib
import sys
import types

from . import config

_ALIAS = "claude_functions"


def module(name: str):
    """Importa ``src/scripts/claude/functions/<name>.py`` sob o alias."""
    if _ALIAS not in sys.modules:
        pkg = types.ModuleType(_ALIAS)
        pkg.__path__ = [str(config.LEGACY_CODE / "functions")]
        sys.modules[_ALIAS] = pkg
    return importlib.import_module(f"{_ALIAS}.{name}")
