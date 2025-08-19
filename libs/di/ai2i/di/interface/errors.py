class DependencyDefinitionError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class OutOfScopeDependencyError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unable to find definition: {name}")


class ManagedInstanceDefinitionError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class CyclicDependecyError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class UnreachableCodeBlockError(Exception):
    def __init__(self) -> None:
        super().__init__("Should never reach here")


class ManagedScopeError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class RoundStorageError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class ProviderBuildError(Exception):
    provider_name: str

    def __init__(self, msg: str, provider_name: str) -> None:
        super().__init__(msg)
        self.provider_name = provider_name


class ProviderReleaseError(Exception):
    provider_name: str

    def __init__(self, msg: str, provider_name: str) -> None:
        super().__init__(msg)
        self.provider_name = provider_name


class ScopeAdapterError(Exception):
    def __init__(self, dependency_name: str) -> None:
        super().__init__(f"Scope adapter failed to resolve dep: '{dependency_name}'")
