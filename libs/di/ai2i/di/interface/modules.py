from __future__ import annotations

from abc import ABC, abstractmethod

from ai2i.di.interface.models import (
    ProvidesDecorator,
    Scope,
)
from ai2i.di.interface.scopes import ProvidersPerScope


class Module(ABC):
    @property
    @abstractmethod
    def providers(self) -> ProvidersPerScope: ...

    @abstractmethod
    def provides(self, *, scope: Scope, name: str | None = None) -> ProvidesDecorator: ...

    @abstractmethod
    def global_init(self) -> ProvidesDecorator: ...
