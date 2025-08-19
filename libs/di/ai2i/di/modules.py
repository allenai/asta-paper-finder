from __future__ import annotations

from typing import Sequence, final

from ai2i.di.interface.models import (
    ProvidesDecorator,
    Scope,
)
from ai2i.di.interface.modules import Module
from ai2i.di.interface.scopes import (
    ProvidersPerScope,
)
from ai2i.di.scopes import (
    ProvidersPerScopeImpl,
)


@final
class ModuleImpl(Module):
    _name: str
    _providers: ProvidersPerScope
    _extends: Sequence[Module]

    def __init__(self, name: str, *, extends: Sequence[Module] = ()) -> None:
        self._name = name
        self._extends = extends

        self._providers = ProvidersPerScopeImpl()
        for m in extends:
            self._providers = self.providers.chain_with(m.providers)

    @property
    def providers(self) -> ProvidersPerScope:
        return self._providers

    def provides(self, *, scope: Scope, name: str | None = None) -> ProvidesDecorator:
        return self._providers.provides(scope=scope, name=name)

    def global_init(self) -> ProvidesDecorator:
        return self._providers.singleton.provides()
