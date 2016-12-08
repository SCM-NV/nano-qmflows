#cython: bounds_check=False

__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============
from libc.math cimport exp, M_PI, sqrt
import numpy as np
cimport numpy as np


# ==================================<>======================================
cpdef double sab_unfolded(np.ndarray r1, np.ndarray r2, str l1, str l2, double c1,
                          double c2, double e1, double e2) except? -1:
    cdef double cte, p, u
    cdef np.ndarray rp  = np.empty(3, dtype=np.float)
    cdef np.ndarray rab = np.empty(3, dtype=np.float)
    cdef np.ndarray rpa = np.empty(3, dtype=np.float)
    cdef np.ndarray rpb = np.empty(3, dtype=np.float)
    cdef np.ndarray rpc = np.empty(3, dtype=np.float)
    cdef np.ndarray s00 = np.empty(3, dtype=np.float)

    cte = sqrt(M_PI / (e1 + e2))
    u = e1 * e2 / (e1 + e2)
    p = 1.0 / (2.0 * (e1 + e2))

    arr3 = np.arange(3, dtype=np.int)
    ls1 = np.apply_along_axis(reverse_calcOrbType, 0, arr3, l1)
    ls2 = np.apply_along_axis(reverse_calcOrbType, 0, arr3, l2)

    rp = (e1 * r1 + e2 * r2) / (e1 + e2)
    rab = r1 - r2
    rpa = rp - r1
    rpb = rp - r2
    s00 = cte * np.exp(-u * rab ** 2.0)

    arr = np.stack([ls1, ls2, np.zeros(3)])
    prod = np.apply_along_axis(obaraSaikaMultipole, 0, arr, p, s00, rpa, rpb, rp) 
    
    return c1 * c2 * prod
    

cpdef double sab(tuple gs1, tuple gs2) except? -1:
    """
    Primitive overlap terms calculated with the Obara-Saika recurrence relations,
    see: Molecular Electronic-Structure Theory. T. Helgaker, P. Jorgensen, J. Olsen. 
    John Wiley & Sons. 2000, pages: 346-347. 

    .. math:: 
        S_{i+1,j} = X_PA * S_{ij} + 1/(2*p) * (i * S_{i-1,j} + j * S_{i,j-1})
        S_{i,j+1} = X_PB * S_{ij} + 1/(2*p) * (i * S_{i-1,j} + j * S_{i,j-1})
    """
    cdef double c1, c2, cte, e1, e2, p, u
    cdef double rab, rp, rpa, rpb, s00, prod = 1
    cdef int i, l1x, l2x
    cdef list r1, r2
    cdef str l1, l2
    
    r1, l1, (c1, e1) = gs1
    r2, l2, (c2, e2) = gs2
    cte = sqrt(M_PI / (e1 + e2))
    u = e1 * e2 / (e1 + e2)
    p = 1.0 / (2.0 * (e1 + e2))
    for i in range(3):
        l1x = calcOrbType_ComponentsC(l1, i)
        l2x = calcOrbType_ComponentsC(l2, i)
        rp = (e1 * r1[i] + e2 * r2[i]) / (e1 + e2)
        rab = r1[i] - r2[i]
        rpa = rp - r1[i]
        rpb = rp - r2[i]
        s00 = cte * exp(-u * rab ** 2.0)
        # select the exponent of the multipole 
        prod *= obaraSaikaMultipole(p, s00, rpa, rpb, rp, l1x, l2x, 0) 
    
    return c1 * c2 * prod

cpdef double sab_efg(tuple gs1, tuple gs2, tuple rc, int e, int f, int g) except? -1:
    """
    Primitive overlap terms calculated with the Obara-Saika recurrence relations,
    see: Molecular Electronic-Structure Theory. T. Helgaker, P. Jorgensen, J. Olsen. 
    John Wiley & Sons. 2000, pages: 346-347. 

    .. math:: 
        S^{e}_{i+1,j} = X_PA * S^{e}_{ij} + 1/(2*p) * (i * S{e}_{i-1,j} + j * S^{e}_{i,j-1} + e * S^{e-1}_{i,j})
        S^{e}_{i,j+1} = X_PB * S^{e}_{ij} + 1/(2*p) * (i * S{e}_{i-1,j} + j * S^{e}_{i,j-1} + e * S^{e-1}_{i,j}
        S^{e+1}_{i,j} = X_PC * S^{e}_{ij} + 1/(2*p) * (i * S{e}_{i-1,j} + j * S^{e}_{i,j-1} + e * S^{e-1}_{i,j})
    """
    cdef double c1, c2, cte, e1, e2, p, u
    cdef double rab, rp, rpa, rpb, rpc, s00, prod = 1
    cdef int i, l1x, l2x
    cdef list r1, r2, multipoles
    cdef str l1, l2

    r1, l1, (c1, e1) = gs1
    r2, l2, (c2, e2) = gs2
    cte = sqrt(M_PI/ (e1 + e2))
    u = e1 * e2 / (e1 + e2)
    p = 1.0 / (2.0 * (e1 + e2))
    multipoles = [e, f, g]

    i = 0 if e != 0 else (1 if f != 0 else 2)

    l1x = calcOrbType_ComponentsC(l1, i)
    l2x = calcOrbType_ComponentsC(l2, i)
    rp = (e1 * r1[i] + e2 * r2[i]) / (e1 + e2)
    rab = r1[i] - r2[i]
    rpa = rp - r1[i]
    rpb = rp - r2[i]
    rpc = rp - rc[i]
    s00 = cte * exp(-u * rab ** 2.0)
    # select the exponent of the multipole 
    prod = obaraSaikaMultipole(p, s00, rpa, rpb, rpc, l1x, l2x, multipoles[i]) 
    
    return c1 * c2 * prod


cdef double obaraSaikaMultipole(double p, double s00x, double xpa, double xpb,
                                double xpc, int i, int j, int e):
    """
    The  Obara-Saika Scheme to calculate overlap integrals. Explicit expressions
    for the s, p, and d orbitals for both de overlap and the integrals
    are written. Higher terms are calculated recursively.
    """
    if i < 0 or j < 0 or e < 0:
        return 0
    elif i == 0 and j == 0 and e == 0:
        return s00x
    elif i == 1 and j == 0 and e == 0:
        return xpa * s00x
    elif i == 0 and j == 1 and e == 0:
        return xpb * s00x
    elif i == 0 and j == 0 and e == 1:
        return xpc * s00x
    elif i == 1 and j == 1 and e == 0:
        return s00x * (xpa * xpb + p)
    elif i == 1 and j == 0 and e == 1:
        return s00x * (xpa * xpc + p)
    elif i == 0 and j == 1 and e == 1:
        return s00x * (xpb * xpc + p)
    elif i == 1 and j == 1 and e == 1:
        return s00x * (xpa * xpb * xpc + p * (xpa + xpb + xpc))
    elif i == 2 and j == 0 and e == 0:
        return s00x * (xpa ** 2 + p)
    elif i == 0 and j == 2 and e == 0:
        return s00x * (xpb ** 2  + p)
    elif i == 2 and j == 0 and e == 1:
        return s00x * ((xpa ** 2) * xpc + p * (2 * xpa + xpc))
    elif i == 0 and j == 2 and e == 1:
        return s00x * ((xpb ** 2) * xpc + p * (2 * xpb + xpc))
    elif i == 2 and j == 1 and e == 0:
        return s00x * ((xpa ** 2) * xpb + p * (2 * xpa + xpb))
    elif i == 1 and j == 2 and e == 0:
        return s00x * (xpa * (xpb ** 2) + p * (xpa + 2 * xpb))
    elif i == 2 and j == 1 and e == 1:
        return s00x * ((xpa ** 2) * xpb * xpc + p * 
                       ((xpa ** 2) + 2 * xpa * xpb + 2 * xpa * xpc + 
                        xpb * xpc + 3 * p ))
    elif i == 1 and j == 2 and e == 1:
        return s00x * (xpa * (xpb ** 2) * xpc + p * 
                       ((xpb ** 2) + 2 * xpa * xpb + 2 * xpb * xpc + 
                        xpa * xpc + 3 * p ))

    # From here on Recursive relations are used.
    elif i >= 1:
        return xpa * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j, e) + \
            p * ((i - 1) * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 2, j, e) +
                 j * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j - 1, e) +
                 e * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j, e - 1))

    elif j >= 1:
        return xpb * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 1, e) + \
            p * (i * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j - 1, e) +
                 (j - 1) * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 2, e) +
                 e * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 1, e - 1))

    elif e >= 1:
        return xpc * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j, e - 1) + \
            p * (i * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i - 1, j, e - 1) +
                 j * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j - 1, e - 1) +
                 (e - 1) * obaraSaikaMultipole(p, s00x, xpa, xpb, xpc, i, j, e - 2))
    

cdef int reverse_calcOrbType(int x, str l):
    """
    Retrieve the cartesian component of a orbital momentum. 
    """
    return orbitalIndexes[l, x]


cdef int calcOrbType_ComponentsC(str l, int x):
    """
    Functions related to the orbital momenta indexes
    """
    return orbitalIndexes[l, x]

cdef dict orbitalIndexes
orbitalIndexes = {
    ("S", 0): 0, ("S", 1): 0, ("S", 2): 0,
    ("Px", 0): 1, ("Px", 1): 0, ("Px", 2): 0,
    ("Py", 0): 0, ("Py", 1): 1, ("Py", 2): 0,
    ("Pz", 0): 0, ("Pz", 1): 0, ("Pz", 2): 1,
    ("Dxx", 0): 2, ("Dxx", 1): 0, ("Dxx", 2): 0,
    ("Dxy", 0): 1, ("Dxy", 1): 1, ("Dxy", 2): 0,
    ("Dxz", 0): 1, ("Dxz", 1): 0, ("Dxz", 2): 1,
    ("Dyy", 0): 0, ("Dyy", 1): 2, ("Dyy", 2): 0,
    ("Dyz", 0): 0, ("Dyz", 1): 1, ("Dyz", 2): 1,
    ("Dzz", 0): 0, ("Dzz", 1): 0, ("Dzz", 2): 2,
    ("Fxxx", 0): 3, ("Fxxx", 1): 0, ("Fxxx", 2): 0,
    ("Fxxy", 0): 2, ("Fxxy", 1): 1, ("Fxxy", 2): 0,
    ("Fxxz", 0): 2, ("Fxxz", 1): 0, ("Fxxz", 2): 1,
    ("Fxyy", 0): 1, ("Fxyy", 1): 2, ("Fxyy", 2): 0,
    ("Fxyz", 0): 1, ("Fxyz", 1): 1, ("Fxyz", 2): 1,
    ("Fxzz", 0): 1, ("Fxzz", 1): 0, ("Fxzz", 2): 2,
    ("Fyyy", 0): 0, ("Fyyy", 1): 3, ("Fyyy", 2): 0,
    ("Fyyz", 0): 0, ("Fyyz", 1): 2, ("Fyyz", 2): 1,
    ("Fyzz", 0): 0, ("Fyzz", 1): 1, ("Fyzz", 2): 2,
    ("Fzzz", 0): 0, ("Fzzz", 1): 0, ("Fzzz", 2): 3
}
