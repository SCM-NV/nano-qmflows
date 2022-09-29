from typing import overload, Any, Literal as L

import numpy as np
from numpy.typing import ArrayLike, NDArray

@overload
def sqrtm(
    A: ArrayLike,
    disp: L[True] = ...,
    blocksize: int = ...,
) -> NDArray[np.floating[Any]]: ...
@overload
def sqrtm(
    A: ArrayLike,
    disp: L[False],
    blocksize: int = ...,
) -> tuple[NDArray[np.floating[Any]], float]: ...
