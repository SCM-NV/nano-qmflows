import types
from typing import Any, Literal as L
from collections.abc import KeysView, MutableMapping, Generator, Sequence

from numpy.typing import DTypeLike, ArrayLike

class AttributeManager(MutableMapping[str, Any]):
    def __setitem__(self, __key: str, __value: ArrayLike) -> None: ...
    def __getitem__(self, __key: str) -> Any: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Generator[str, None, None]: ...
    def __delitem__(self, __key: str) -> None: ...

class Group(MutableMapping[str, Any]):
    def __getitem__(self, key: str) -> Any: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Generator[str, None, None]: ...
    def __setitem__(self, __key: str, __value: Any) -> None: ...
    def __delitem__(self, __key: str) -> None: ...
    def require_dataset(self, name: str, shape: Sequence[int], dtype: DTypeLike, exact: bool = ..., **kwds: Any) -> Any: ...
    def keys(self) -> KeysView[str]: ...
    def move(self, source: str, dest: str) -> None: ...
    @property
    def attrs(self) -> AttributeManager: ...

class File(Group):
    def __init__(
        self,
        name,
        mode: L["r", "r+", "w", "w-", "x", "a"] = ...,
    ) -> None: ...
    def __enter__(self) -> File: ...
    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_val: BaseException | None,
        __exc_tb: types.TracebackType | None,
    ) -> None: ...
