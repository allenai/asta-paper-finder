from contextlib import nullcontext
from typing import Any, Callable, cast, final

from ai2i.common.utils.value import ValueNotSet
from ai2i.config import ConfigValue, ConfigValuePlaceholder
from ai2i.di.context import ResolverFromContextVar
from ai2i.di.interface.models import (
    DefaultFactory,
    DependencyDefinition,
    DependencyPlaceholder,
)
from ai2i.di.resolver import GetDependencyOperations, ManagedOperations
from ai2i.di.scopes import ScopeAdaptingDynamicProxy


@final
class DIGateway(ManagedOperations, GetDependencyOperations):
    """
    A small utility function to be used by both the ManagedEnv and ApplicationContext.
    This allows the environemnts to define a requirement for a parameter of a managed
    function or a provider:

    Examples:
      1. Providers
      ```
        @env.provides()
        async def my_value(other_value: OtherValue = DI.require(other)) -> MyValue:
            return await MyValue.build(other_value.val)
      ```
      2. Managed
      ```
        @env.managed
        async def add_other(v: int, other_value: OtherValue = DI.require(other)) -> int:
            return await other_value.val + v
      ```

    """

    def __init__(self) -> None:
        super().__init__(ResolverFromContextVar())

    def requires[A, B](
        self,
        definition: DependencyDefinition[A],
        /,
        default: B | ValueNotSet = ValueNotSet.instance(),
        default_factory: DefaultFactory[A] | ValueNotSet = ValueNotSet.instance(),
    ) -> A | B:
        placeholder = DependencyPlaceholder(definition, default, default_factory)
        return cast(A | B, placeholder)

    def requires_by_name(
        self,
        name: str,
        /,
        default: Any | ValueNotSet = ValueNotSet.instance(),
        default_factory: DefaultFactory[Any] | ValueNotSet = ValueNotSet.instance(),
    ) -> Any:
        return DependencyPlaceholder(
            # NOTE: for require the other arguments, beside name can be ignored
            DependencyDefinition(name, lambda _: nullcontext(), []),
            default,
            default_factory,
        )

    def config[A, B](
        self,
        reader: ConfigValuePlaceholder[A],
        /,
        default: B | ValueNotSet = ValueNotSet.instance(),
        default_factory: Callable[[], B] | ValueNotSet = ValueNotSet.instance(),
    ) -> A | B:
        return ConfigValue(reader, default=default, default_factory=default_factory)

    def scope_adapting_dynamic_proxy_dep[A](self, dep: DependencyDefinition[A]) -> A:
        """
        get a dynamic proxy that will proxy all calls to a dependency based on the
        currently available entity in scope
        """
        return cast(A, ScopeAdaptingDynamicProxy(dep))


DI = DIGateway()
