from collections.abc import Callable
from typing import Any
from typing_extensions import ParamSpec

_P = ParamSpec("_P")

def schedule(f: Callable[_P, object], **hints: Any) -> Callable[_P, PromisedObject]: ...

class PromisedObject(Any):
    def __init__(self, workflow: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> PromisedObject: ...
    def __getattr__(self, attr: str) -> PromisedObject: ...
    def __setattr__(self, attr: str, value: Any) -> None: ...
    def __result__(self) -> Any: ...
    def __lt__(self, other) -> PromisedObject: ...
    def __gt__(self, other) -> PromisedObject: ...
    def __eq__(self, other) -> PromisedObject: ...
    def __ne__(self, other) -> PromisedObject: ...
    def __ge__(self, other) -> PromisedObject: ...
    def __le__(self, other) -> PromisedObject: ...
    def __bool__(self) -> PromisedObject: ...
    def __abs__(self) -> PromisedObject: ...
    def __sub__(self, other) -> PromisedObject: ...
    def __add__(self, other) -> PromisedObject: ...
    def __mul__(self, other) -> PromisedObject: ...
    def __rmul__(self, other) -> PromisedObject: ...
    def __truediv__(self, other) -> PromisedObject: ...
    def __floordiv__(self, other) -> PromisedObject: ...
    def __mod__(self, other) -> PromisedObject: ...
    def __pow__(self, other) -> PromisedObject: ...
    def __pos__(self) -> PromisedObject: ...
    def __neg__(self) -> PromisedObject: ...
    def __matmul__(self, other) -> PromisedObject: ...
    def __index__(self) -> PromisedObject: ...
    def __inv__(self) -> PromisedObject: ...
    def __lshift__(self, n) -> PromisedObject: ...
    def __rshift__(self, n) -> PromisedObject: ...
    def __and__(self, other) -> PromisedObject: ...
    def __or__(self, other) -> PromisedObject: ...
    def __xor__(self, other) -> PromisedObject: ...
    def __contains__(self, item: Any) -> PromisedObject: ...
    def __getitem__(self, name: Any) -> PromisedObject: ...
    def __setitem__(self, attr: str, value: Any) -> None: ...
