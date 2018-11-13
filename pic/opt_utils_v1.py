import numpy as np
from copy import deepcopy
import time

# ...
import utilities.__epyccel__core as mod
mod = mod.epyccel__core
core_fieldInterpolation_bc_1 = mod.fieldinterpolation_bc_1
core_hotCurrent_bc_1 = mod.hotcurrent_bc_1
core_borisPush_bc_1 = mod.borispush_bc_1
core_new_borisPush_bc_1 = mod.new_borispush_bc_1
# ...

# ...
def fieldInterpolation_bc_1(particles_pos, knots, p, Nb, Lz, ex, ey, bx, by, Ep, Bp):

    Ep[:,:2] = 0.
    Bp[:,:2] = 0.

    core_fieldInterpolation_bc_1(particles_pos, knots, p, Nb, ex, ey, bx, by, Ep, Bp)
# ...

# ...
def hotCurrent_bc_1(particles_vel, particles_pos, particles_wk, knots, p, Nb, qe, c, jh):

    jh[:] = 0.

    core_hotCurrent_bc_1(particles_vel, particles_pos, particles_wk, knots, p, Nb, qe, c, jh)
# ...

# ...
def borisPush_bc_1(particles, dt, B, E, qe, me, Lz):
    core_borisPush_bc_1(particles, dt, B, E, qe, me, Lz)
# ...

# ...
def new_borisPush_bc_1(particles, dt, qe, me, Lz, knots, p, Nb, ex, ey, bx, by, B0z):
    core_new_borisPush_bc_1(particles, dt, qe, me, Lz, knots, p, Nb, ex, ey, bx, by, B0z)
# ...
