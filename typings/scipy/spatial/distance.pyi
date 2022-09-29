from typing import overload, Any, Protocol, Literal, SupportsIndex, SupportsFloat

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MetricKind = Literal[
    'braycurtis',
    'canberra',
    'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch',
    'cityblock', 'cblock', 'cb', 'c',
    'correlation', 'co',
    'cosine', 'cos',
    'dice',
    'euclidean', 'euclid', 'eu', 'e',
    'hamming', 'hamm', 'ha', 'h',
    'minkowski', 'mi', 'm', 'pnorm',
    'jaccard', 'jacc', 'ja', 'j',
    'jensenshannon', 'js',
    'kulsinski', 'kulczynski1',
    'mahalanobis', 'mahal', 'mah',
    'rogerstanimoto',
    'russellrao',
    'seuclidean', 'se', 's',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean', 'sqe', 'sqeuclid',
    'yule',
]

class _MetricCallback1(Protocol):
    def __call__(
        self, __XA: NDArray[Any], __XB: NDArray[Any]
    ) -> None | str | bytes | SupportsFloat | SupportsIndex: ...

class _MetricCallback2(Protocol):
    def __call__(
        self, __XA: NDArray[Any], __XB: NDArray[Any], **kwargs: Any
    ) -> None | str | bytes | SupportsFloat | SupportsIndex: ...

@overload
def cdist(
    XA: ArrayLike,
    XB: ArrayLike,
    metric: _MetricKind = ...,
    *,
    out: None | NDArray[np.floating[Any]] = ...,
    p: float = ...,
    w: None | ArrayLike = ...,
    V: None | ArrayLike = ...,
    VI: None | ArrayLike = ...,
) -> NDArray[np.floating[Any]]: ...
@overload
def cdist(
    XA: ArrayLike,
    XB: ArrayLike,
    metric: _MetricCallback1 | _MetricCallback2,
    *,
    out: None | NDArray[np.floating[Any]] = ...,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]: ...
