from typing import Sequence

from ai2i.di.interface.modules import Module
from ai2i.di.modules import ModuleImpl


def create_module(name: str, *, extends: Sequence[Module] = ()) -> Module:
    return ModuleImpl(name, extends=extends)
