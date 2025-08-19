from __future__ import annotations

# a secret value only known to this module to prevent creating multpile instances of ValueNotSet
import uuid

_secret = str(uuid.uuid4())


class ValueNotSet:
    """
    simple utility to indicate a value not set, for cases where None is a valid
    value and can not be used as the "NotSet" value

    The only valid value is: `ValueNotSet.instance()`
    """

    @staticmethod
    def instance() -> ValueNotSet:
        return _VALUE_NOT_SET

    def __init__(self, secret: str) -> None:
        if _secret != secret:
            raise AssertionError("Can not create such instance, use the singleton ValueNotSet.instance()")

    def __repr__(self) -> str:
        return "ValueNotSet"


_VALUE_NOT_SET = ValueNotSet(_secret)
