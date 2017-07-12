

__author__ = "Felipe Zapata"

# ==========> Standard libraries and third-party <===============
cimport cython
from cpython cimport bool # Python Boolean
from libc.math cimport exp, log, M_PI, sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
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
    rab = distance(r1, r2)

    if neglect_integral(rab, e1, e2, 1e-10):
        return 0
    else:
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


@cython.boundscheck(False)
@cython.wraparound(False)
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
    cdef list r1, r2
    cdef str l1, l2
    cdef list multipoles = [e, f, g]

    r1, l1, (c1, e1) = gs1
    r2, l2, (c2, e2) = gs2
    cte = sqrt(M_PI/ (e1 + e2))
    u = e1 * e2 / (e1 + e2)
    p = 1.0 / (2.0 * (e1 + e2))

    for i in range(3):
        l1x = calcOrbType_ComponentsC(l1, i)
        l2x = calcOrbType_ComponentsC(l2, i)
        rp = (e1 * r1[i] + e2 * r2[i]) / (e1 + e2)
        rab = r1[i] - r2[i]
        rpa = rp - r1[i]
        rpb = rp - r2[i]
        rpc = rp - rc[i]
        s00 = cte * exp(-u * rab ** 2.0)
        # select the exponent of the multipole 
        prod *= obaraSaikaMultipole(p, s00, rpa, rpb, rpc, l1x, l2x, multipoles[i]) 
    
    return c1 * c2 * prod


cpdef double obaraSaikaMultipole(double p, double s00x, double xpa, double xpb,
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool neglect_integral(double r, double e1, double e2, double accuracy):
    """
    Compute whether an overlap integral should be neglected 
    """
    cdef double a, ln
    a = min(e1, e2)
    ln = log(((M_PI / (2 * a)) ** 3) * 10 ** (2 * accuracy))
    
    # Check if the condition is fulfill
    b = r > sqrt((1 / a) * ln) 

    return b

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double distance(list xs, list ys):
    """
    Distance between 2 points
    """
    cdef double acc=0, x, y
    for x,y in zip(xs, ys):
        acc += (x - y) ** 2

    return sqrt(acc)
