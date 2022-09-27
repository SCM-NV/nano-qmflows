"""Monkey patch noodles with support for slots-containing dataclasses."""

import dataclasses

from noodles.serial.dataclass import SerDataClass
from scm.plams import add_to_class


@add_to_class(SerDataClass)
def encode(self, obj, make_rec):
    """Encode the passed dataclass."""
    if hasattr(obj, "__dict__"):
        return make_rec(obj.__dict__)
    else:
        return make_rec(dataclasses.asdict(obj))
