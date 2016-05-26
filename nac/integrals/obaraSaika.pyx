#cython: bounds_check=False

__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============
from libc.math cimport exp, sqrt
from math import pi

# ==================================<>======================================

cpdef double sab(tuple gs1, tuple gs2) except? -1:
    """
    Primitive overlap terms calculated with the Obara-Saika recurrence relations,
    see: Molecular Electronic-Structure Theory. T. Helgaker, P. Jorgensen, J. Olsen. 
    John Wiley & Sons. 2000, pages: 345-346. 

    .. math:: 
        s_i+1,j = X_PA * S_ij + 1 /(2*p) * (i * S_i-1,j + j * S_i,j-1)     
        s_i,j+1 = X_PB * S_ij + 1 /(2*p) * (i * S_i-1,j + j * S_i,j-1)     
    

    """
    cdef double c1, c2, cte, e1, e2, p, u
    cdef double rab, rp, rpa, rpb, s00, prod = 1
    cdef int i, l1x, l2x
    cdef list r1, r2
    cdef str l1, l2

    r1, l1, (c1, e1) = gs1
    r2, l2, (c2, e2) = gs2
    cte = sqrt(pi / (e1 + e2))
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
        prod *= obaraSaika(p, s00,l1x, l2x, rpa, rpb) 
    
    return c1 * c2 * prod


cdef double obaraSaika(double p, double s00x, int i, int j, double xpa, double xpb):
    """
    The  Obara-Saika Scheme to calculate overlap integrals
    """
    if i < 0 or j < 0:
        return 0
    elif i == 0 and j == 0:
        return s00x
    elif i == 1 and j == 0:
        return xpa * s00x
    elif i == 0 and j == 1:
        return xpb * s00x
    elif i == 1 and j == 1:
        return s00x * (xpa * xpb + p)
    elif i == 2 and j == 0:
        return s00x * (xpa ** 2 + p)
    elif i == 0 and j == 2:
        return s00x * (xpb ** 2  + p)
    elif i == 2 and j == 1:
        return s00x * ((xpa ** 2) * xpb + p * (2 * xpa + xpb))
    elif i == 1 and j == 2:
        return s00x * (xpa * (xpb ** 2) + p * (xpa + 2 * xpb))
    elif i >= 1:
        return xpa * (obaraSaika(p, s00x, i - 1, j, xpa, xpb)) + \
            p * ((i - 1) * (obaraSaika(p, s00x, i - 2, j, xpa, xpb)) +
                 j * (obaraSaika(p, s00x, i - 1, j - 1, xpa, xpb)))

    elif j >= 1:
        return xpb * (obaraSaika(p, s00x, i, j - 1, xpa, xpb)) + \
            p * (i * (obaraSaika(p, s00x, i - 1, j - 1, xpa, xpb)) +
                 (j - 1) * (obaraSaika(p, s00x, i, j - 2, xpa, xpb)))


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
