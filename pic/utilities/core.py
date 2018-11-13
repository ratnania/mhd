from pyccel.decorators import types
from pyccel.decorators import pure

from bsplines import find_span
from bsplines import basis_funs

#==============================================================================
@types('double[:]','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','double[:,:](order=F)')
def fieldInterpolation_bc_1(particles_pos, knots, p, Nb, ex, ey, bx, by, Ep, Bp):

    from numpy      import empty
    from numpy      import zeros

    left   = empty( p  , dtype=float )
    right  = empty( p  , dtype=float )
    values = zeros( p+1, dtype=float )

    # to make it inout
    span = 0

    # ...
    npart = len(particles_pos)
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, left, right, values )

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            Ep[ipart, 0] += ex[ii]*bi
            Ep[ipart, 1] += ey[ii]*bi
            Bp[ipart, 0] += bx[ii]*bi
            Bp[ipart, 1] += by[ii]*bi
    # ...
