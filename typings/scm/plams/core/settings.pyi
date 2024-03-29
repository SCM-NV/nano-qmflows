import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Generic, TypeVar, overload, Protocol

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_MT = TypeVar("_MT", bound=_SupportsMissing)
_ST = TypeVar("_ST", bound=Settings[Any, Any])

class _SupportsMissing(Protocol):
    def __missing__(self, __key: Any) -> Any: ...

class Settings(dict[_KT, _VT]):
    def copy(self: _ST) -> _ST: ...
    def soft_update(self: _ST, other: Mapping[_KT, _VT]) -> _ST: ...
    def update(self, other: Mapping[_KT, _VT]) -> None: ...  # type: ignore[override]
    def merge(self: _ST, other: Mapping[_KT, _VT]) -> _ST: ...
    def find_case(self, key: _KT) -> _KT: ...
    def as_dict(self) -> dict[_KT, _VT]: ...
    @classmethod
    def suppress_missing(cls: type[_MT]) -> SuppressMissing[_MT]: ...
    def get_nested(self, key_tuple: Iterable[Any], suppress_missing: bool = False) -> Any: ...
    def set_nested(self, key_tuple: Sequence[Any], value: Any, suppress_missing: bool = False) -> None: ...
    def flatten(self, flatten_list: bool = ...) -> Settings[tuple[Any, ...], Any]: ...
    def unflatten(self, unflatten_list: bool = ...) -> Settings[Any, Any]: ...
    @classmethod  # type: ignore[override]
    @overload
    def fromkeys(cls, __iterable: Iterable[_KT]) -> Settings[_KT, Any]: ...
    @classmethod
    @overload
    def fromkeys(cls, __iterable: Iterable[_KT], __value: _VT) -> Settings[_KT, _VT]: ...
    def __missing__(self, __key: _KT) -> Settings[Any, Any]: ...
    def __getattr__(self, name: _KT) -> _VT: ...  # type: ignore[misc]
    def __setattr__(self, name: _KT, value: _VT) -> None: ...  # type: ignore[misc,override]
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __add__(self: _ST, other: Mapping[_KT, _VT]) -> _ST: ...
    def __iadd__(self: _ST, other: Mapping[_KT, _VT]) -> _ST: ...
    def __copy__(self: _ST) -> _ST: ...

class SuppressMissing(Generic[_MT]):
    obj: _MT
    missing: Callable[[Any, _MT, Any], Any]
    def __init__(self, obj: type[_MT]) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self, exc_type: None | type[BaseException], exc_value: None | BaseException, traceback: None | types.TracebackType
    ) -> None: ...
