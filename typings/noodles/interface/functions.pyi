from typing import TypeVar, Protocol, SupportsIndex
from collections.abc import Generator

from .decorator import schedule

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

class _SupportsGetItem(Protocol[_T_co]):
    def __getitem__(self, __key: SupportsIndex) -> _T_co: ...

@schedule
def gather(*a: _T) -> list[_T]: ...
def unpack(t: _SupportsGetItem[_T], n: SupportsIndex) -> Generator[_T, None, None]: ...
