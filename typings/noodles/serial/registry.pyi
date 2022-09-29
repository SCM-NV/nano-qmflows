import abc
from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")


class Serialiser(abc.ABC):
    name: str
    def __init__(self, name: str | Callable[..., Any] = ...) -> None: ...
    @abc.abstractmethod
    def encode(self, obj: Any, make_rec: Callable[[Any], _T]) -> _T: ...
    @abc.abstractmethod
    def decode(self, cls: type[_T], data: Any) -> _T: ...
