import numpy as np
from copy import deepcopy
from bsplines import find_span, basis_funs


# ... NOTE: this function can not be pyccelized for the moment
def find_tiles(particles_pos, Nz, Lz, tilesize):
    d_tiles = {}
    for ie in range(0, Nz, tilesize):
        d_tiles[ie] = []

    npart = len(particles_pos)
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        i = int(Nz*pos/Lz)
        ie = int(i/tilesize)*tilesize
        d_tiles[ie] += [ipart]

    return d_tiles
# ...


# ... NOTE: this function can not be pyccelized for the moment
def fieldInterpolation_bc_1(particles_pos, knots, p, Nb, Lz, ex, ey, bx, by, values):
    tilesize = p
    d_tiles = find_tiles(particles_pos, Nb, Lz, tilesize)

    Ep = np.zeros((len(particles_pos), 2))
    Bp = np.zeros((len(particles_pos), 2))

    # ...
    for ie, ipositions in d_tiles.items():
        positions = deepcopy(particles_pos[ipositions])
        core_fieldInterpolation_bc_1(positions, ipositions, knots, p, Nb, Lz, ex, ey, bx, by, values, Ep, Bp)
    # ...

    return Ep, Bp
# ...

def core_fieldInterpolation_bc_1(positions, ipositions, knots, p, Nb, Lz, ex, ey, bx,
                                 by, values, Ep, Bp):

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
