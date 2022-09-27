from collections.abc import Callable
from typing import Any, TypeVar

_FT = TypeVar("_FT", bound=Callable[..., Any])

def add_to_class(classname: type[Any]) -> Callable[[_FT], _FT]: ...
