from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, ParamSpec, Sequence, TypeVar

P = ParamSpec("P")
P2 = ParamSpec("P2")

B = TypeVar("B")
B2 = TypeVar("B2")


@dataclass(frozen=True)
class Substitution:
    placeholder_type: type[Any]
    resolve: Callable[[Any], Any]


@dataclass(frozen=True)
class SubstitutionInfo(Generic[P, B]):
    arg_names: Sequence[str]
    args_and_defaults: Mapping[str, Any]

    kwarg_names: Sequence[str]
    kwargs_and_defaults: Mapping[str, Any]

    @staticmethod
    def from_function(f: Callable[P2, B2]) -> SubstitutionInfo[P2, B2]:
        f_spec = inspect.getfullargspec(f)
        arg_names = f_spec.args

        arg_defaults = f_spec.defaults

        if arg_defaults is None:
            arg_defaults = ()

        args_and_defaults = {t[0]: t[1] for t in list(zip(reversed(arg_names), reversed(list(arg_defaults))))}

        kwarg_names = f_spec.kwonlyargs
        kwargs_and_defaults = f_spec.kwonlydefaults

        if kwargs_and_defaults is None:
            kwargs_and_defaults = {}

        return SubstitutionInfo(arg_names, args_and_defaults, kwarg_names, kwargs_and_defaults)

    def resolve(
        self, substitutions: Sequence[Substitution], *args: P.args, **kwargs: P.kwargs
    ) -> tuple[Sequence[Any], Mapping[str, Any]]:
        resolved_args: list[Any] = []
        for args_pos, arg_name in enumerate(self.arg_names):
            if args_pos < len(args):
                resolved_args.append(args[args_pos])
            elif arg_name in kwargs:
                resolved_args.append(kwargs[arg_name])
            elif arg_name in self.args_and_defaults:
                default_value = self.args_and_defaults[arg_name]
                has_substitution = False
                for sub in substitutions:
                    if isinstance(default_value, sub.placeholder_type):
                        resolved_args.append(sub.resolve(default_value))
                        has_substitution = True
                        break
                if not has_substitution:
                    resolved_args.append(default_value)
            else:
                raise AssertionError("should never reach here")

        resolved_kwargs: dict[str, Any] = {}
        for kwarg_name in self.kwarg_names:
            if kwarg_name in kwargs:
                resolved_kwargs[kwarg_name] = kwargs[kwarg_name]
            elif kwarg_name in self.kwargs_and_defaults:
                default_value = self.kwargs_and_defaults[kwarg_name]
                has_substitution = False
                for sub in substitutions:
                    if isinstance(default_value, sub.placeholder_type):
                        resolved_kwargs[kwarg_name] = sub.resolve(default_value)
                        has_substitution = True
                        break
                if not has_substitution:
                    resolved_kwargs[kwarg_name] = default_value
            else:
                raise AssertionError("should never reach here")

        return resolved_args, resolved_kwargs
