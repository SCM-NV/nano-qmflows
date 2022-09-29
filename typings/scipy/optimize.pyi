from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

def curve_fit(
    f: Callable[..., Any],
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: None | ArrayLike = ...,
    sigma: None | ArrayLike = ...,
    absolute_sigma: bool = ...,
    check_finite: bool = ...,
    bounds: tuple[ArrayLike, ArrayLike] = ...,
    method: None | str = ...,
    jac: Callable[..., Any] | str | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]: ...

def linear_sum_assignment(
    cost_matrix: ArrayLike,
    maximize: bool = ...,
) -> NDArray[np.intp]: ...
