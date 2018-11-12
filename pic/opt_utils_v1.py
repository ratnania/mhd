import numpy as np
from copy import deepcopy
import time

VERSION = 0

# ...
import utilities.__epyccel__core as mod
mod = mod.epyccel__core
if VERSION == 0:
    core_fieldInterpolation_bc_1 = mod.fieldinterpolation_bc_1_v0
if VERSION == 1:
    core_fieldInterpolation_bc_1 = mod.fieldinterpolation_bc_1_v1
# ...

# ...
def fieldInterpolation_bc_1_v0(particles_pos, knots, p, Nb, Lz, ex, ey, bx, by,
                            values, Ep, Bp):

    Ep[:,:2] = 0.
    Bp[:,:2] = 0.

    core_fieldInterpolation_bc_1(particles_pos, knots, p, Nb, ex, ey, bx, by, values, Ep, Bp)
# ...

# ...
def fieldInterpolation_bc_1_v1(particles_pos, knots, p, Nb, Lz, ex, ey, bx, by,
                            values, Ep, Bp):

    # ...
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

    tilesize = 10
    tb = time.time()
    d_tiles = find_tiles(particles_pos, Nb, Lz, tilesize)
    te = time.time()
    print('> time tiling: ' + str(te - tb))

    tb = time.time()

    Ep[:,:2] = 0.
    Bp[:,:2] = 0.

    # ...
    for ie, ipositions in d_tiles.items():
        positions = deepcopy(particles_pos[ipositions])
        core_fieldInterpolation_bc_1(positions, ipositions, knots, p, Nb, Lz, ex, ey, bx, by, values, Ep, Bp)
    # ...
    te = time.time()
    print('> time interpolation: ' + str(te - tb))
# ...

if VERSION == 0:
    fieldInterpolation_bc_1 = fieldInterpolation_bc_1_v0
if VERSION == 1:
    fieldInterpolation_bc_1 = fieldInterpolation_bc_1_v1

