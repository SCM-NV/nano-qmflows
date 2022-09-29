import dataclasses
from typing import TypeVar, Any
from collections.abc import Callable
from noodles.serial import Serialiser

_DataClass = Any
_T = TypeVar("_T")
_Self = TypeVar("_Self", bound=_DataClass)


class SerDataClass(Serialiser):
    def __init__(self) -> None: ...
    def encode(self, obj: _DataClass, make_rec: Callable[[dict[str, Any]], _T]) -> _T: ...
    def decode(self, cls: type[_Self], data: dict[str, Any]) -> _Self: ...
