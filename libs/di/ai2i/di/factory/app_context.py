from ai2i.di.app_context import ApplicationContextImpl, builtin_module
from ai2i.di.interface.app_context import ApplicationContext
from ai2i.di.interface.modules import Module


def create_app_context(module: Module) -> ApplicationContext:
    return ApplicationContextImpl(module)


def create_empty_app_context() -> ApplicationContext:
    return ApplicationContextImpl(builtin_module)
