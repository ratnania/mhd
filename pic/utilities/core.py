from pyccel.decorators import types
from pyccel.decorators import pure

from bsplines import find_span
from bsplines import basis_funs

#==============================================================================
@types('double[:]','int[:]','double[:]','int','int','double','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','double[:,:](order=F)')
def fieldInterpolation_bc_1_v1(positions, ipositions, knots, p, Nb, Lz, ex, ey, bx, by, values, Ep, Bp):

    # to make it inout
    values[:] = 0.
    span = 0

    npart = len(positions)
    for ilpart in range(0, npart):
        ipart = ipositions[ilpart]
        pos = positions[ilpart]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, values )

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            Ep[ipart, 0] += ex[ii]*bi
            Ep[ipart, 1] += ey[ii]*bi
            Bp[ipart, 0] += bx[ii]*bi
            Bp[ipart, 1] += by[ii]*bi

#==============================================================================
@types('double[:]','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','double[:,:](order=F)')
def fieldInterpolation_bc_1_v0(particles_pos, knots, p, Nb, ex, ey, bx, by, values, Ep, Bp):

    # to make it inout
    values[:] = 0.
    span = 0

    # ...
    npart = len(particles_pos)
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, values )

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            Ep[ipart, 0] += ex[ii]*bi
            Ep[ipart, 1] += ey[ii]*bi
            Bp[ipart, 0] += bx[ii]*bi
            Bp[ipart, 1] += by[ii]*bi
    # ...
