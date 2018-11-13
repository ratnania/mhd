import numpy as np
from copy import deepcopy
import time

VERSION = 0

# ...
import utilities.__epyccel__core as mod
mod = mod.epyccel__core
core_fieldInterpolation_bc_1 = mod.fieldinterpolation_bc_1
# ...

# ...
def fieldInterpolation_bc_1(particles_pos, knots, p, Nb, Lz, ex, ey, bx, by,
                            Ep, Bp):

    Ep[:,:2] = 0.
    Bp[:,:2] = 0.

    core_fieldInterpolation_bc_1(particles_pos, knots, p, Nb, ex, ey, bx, by, Ep, Bp)
# ...
