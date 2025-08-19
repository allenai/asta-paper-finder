from contextvars import ContextVar

from ai2i.di.interface.errors import OutOfScopeDependencyError
from ai2i.di.interface.models import DependencyDefinition
from ai2i.di.interface.scopes import ApplicationScopes

ctx_active_scopes: ContextVar[ApplicationScopes | None] = ContextVar("_ctx_active_scopes", default=None)


class ResolverFromContextVar:
    def __call__[A](self, definition: DependencyDefinition[A]) -> A:
        active_scopes = ctx_active_scopes.get()
        if active_scopes is None:
            raise OutOfScopeDependencyError("No application context found to resolve from")
        return active_scopes.resolve(definition)
