
# coding: utf-8

# This is a version of the electron hybrid code with some extracted pieces.
# The most time consuming functions are utils.FieldInterpolation, utils.hotCurrent and utils.borisPush

# In[25]:


import numpy as np
import time
from copy import deepcopy
from scipy.linalg import block_diag
import utils
import opt_utils_v0
import opt_utils_v1


# ... physical parameters
eps0 = 1.0                         # ... vacuum permittivity
mu0 = 1.0                          # ... vacuum permeability
c = 1.0                            # ... speed of light
qe = -1.0                          # ... electron charge
me = 1.0                           # ... electron mass
B0z = 1.0                          # ... background magnetic field in z-direction
wce = qe*B0z/me                    # ... electron cyclotron frequency
wpe = 2*np.abs(wce)                # ... cold electron plasma frequency
nuh = 6e-2                         # ... ratio of cold/hot electron densities (nh/nc)
nh = nuh*wpe**2                    # ... hot electron density
wpar = 0.2*c                       # ... parallel thermal velocity of energetic particles
wperp = 0.2*c                      # ... perpendicular thermal velocity of energetic particles
# ...



# ... parameters for initial conditions
k = 2                              # ... wavenumber of initial wave fields
ini = 3                            # ... initial conditions for wave fields
amp = 1e-1                         # ... amplitude of initial wave fields
eps = 0.0                          # ... amplitude of spatial pertubation of distribution function
# ...



# ... numerical parameters
Lz = 300                           # ... length of z-domain
#Nz = 1024                          # ... number of elements z-direction
#Nz = 128                          # ... number of elements z-direction
Nz = 32                          # ... number of elements z-direction
T = 200.0                          # ... simulation time
dt = 0.05                          # ... time step
p = 3                              # ... degree of B-spline basis
Lv = 8                             # ... length of v-domain in each direction (vx,vy,vz)
Nv = 76                            # ... number of cells in each v-direction (vx,vy,vz)
Np = np.int(1e6)                   # ... number of energetic simulation particles
#Np = np.int(1e4)                   # ... number of energetic simulation particles
# ...



# ... discretization parameters
dz = Lz/Nz
zj = np.linspace(0, Lz, Nz + 1)

dv = Lv/Nv
vj = np.linspace(-Lv/2, Lv/2, Nv+1)
# ...




# ... system matrices for fluid electrons and electromagnetic fields
A10 = np.array([0, 0, 0, +c**2, 0, 0])
A11 = np.array([0, 0, -c**2, 0, 0, 0])
A12 = np.array([0, -1, 0, 0, 0, 0])
A13 = np.array([+1, 0, 0, 0, 0, 0])
A14 = np.array([0, 0, 0, 0, 0, 0])
A15 = np.array([0, 0, 0, 0, 0, 0])
A1 = np.array([A10, A11, A12, A13, A14, A15])

A20 = np.array([0, 0, 0, 0, mu0*c**2, 0])
A21 = np.array([0, 0, 0, 0, 0, mu0*c**2])
A22 = np.array([0, 0, 0, 0, 0, 0])
A23 = np.array([0, 0, 0, 0, 0, 0])
A24 = np.array([-eps0*wpe**2, 0, 0, 0, 0, -wce])
A25 = np.array([0, -eps0*wpe**2, 0, 0, +wce, 0])
A2 = np.array([A20, A21, A22, A23, A24, A25])

s = int(np.sqrt(A1.size))

def B_background_z(z):
    return B0z*(1 + 0*(z - Lz/2)**2)
# ...



# ... create periodic B-spline basis and quadrature grid
bsp, N, quad_points, weights = utils.createBasis(Lz, Nz, p)
# ...





# ... matrices for linear system
Nb = N - p                        # ... number of unique B-splines for periodic boundary conditions

uj = np.zeros(s*Nb)               # ... coefficients for Galerkin approximation
Fh = np.zeros(s*Nb)               # ... RHS of matrix system

Mblock = np.zeros((s*Nb, s*Nb))   # ... block mass matrix
Cblock = np.zeros((s*Nb, s*Nb))   # ... block convection matrix

u0 = np.zeros((Nb, s))            # ... L2-projection of initial conditions

A1block = block_diag(*([A1]*Nb))  # ... block system matrix A1
A2block = block_diag(*([A2]*Nb))  # ... block system matrix A2
# ...





# ... assemble mass, advection and field matrices
timea = time.time()

M, C, D = utils.matrixAssembly(bsp, weights, quad_points, B_background_z, 1)

timeb = time.time()
print('time for matrix assembly: ' + str(timeb - timea))
# ...




# ... assemble u0
timea = time.time()

for qu in range (0, s):

    def initial(z):
        return utils.IC(z, ini, amp, k, omega = 0)[qu]

    u0[:,qu] = utils.L2proj(bsp, Lz, quad_points, weights, M, initial)

uj = np.reshape(u0, s*Nb)

timeb = time.time()
print('time for initial vector assembly: ' + str(timeb - timea))
# ...



# ... create particles (z, vx, vy, vz, wk) and sample positions and velocities according to sampling distribution
particles = np.zeros((Np, 5))
particles[:, 0] = np.random.rand(Np)*Lz
particles[:, 1] = np.random.randn(Np)*wperp
particles[:, 2] = np.random.randn(Np)*wperp
particles[:, 3] = np.random.randn(Np)*wpar
particles[:, 4] = np.random.rand(Np)
# ...





# ---------------------------------!to be accelerated!----------------------------------------------------
# ------------------------------------------------------------------------------------------------------



# ... initial fields at particle positions
Ep = np.zeros((Np, 3))
Bp = np.zeros((Np, 3))
Bp[:, 2] = B0z

timea = time.time()

Ep[:, 0:2], Bp[:, 0:2] = utils.fieldInterpolation(particles[:, 0], zj, bsp, uj)

timeb = time.time()
print('time for intial field interpolation: ' + str(timeb - timea))

# ... ARA
_Ep = np.zeros((Np, 3), order='F')
_Bp = np.zeros((Np, 3), order='F')
_Bp[:, 2] = B0z

ex = uj[0::6]
ey = uj[1::6]
bx = uj[2::6]
by = uj[3::6]

knots = bsp.T
p = bsp.p

timea = time.time()
opt_utils_v1.fieldInterpolation_bc_1(particles[:, 0],
                                     knots, p, Nz, Lz,
                                     ex, ey,
                                     bx, by,
                                     _Ep,_Bp)
timeb = time.time()
print('time for new field interpolation: ' + str(timeb - timea))

assert(np.allclose(Ep, _Ep))
assert(np.allclose(Bp, _Bp))
_particles = np.zeros(particles.shape, order='F')
_particles[:,:] = particles[:,:]
# ...

# ... initialize velocities by pushing back by -dt/2, compute weights and energy of hot particles
timea = time.time()

particles[:, 1:4] = utils.borisPush(particles, -dt/2, Bp, Ep, qe, me, Lz)[1]

timeb = time.time()
print('time for intial particle push: ' + str(timeb - timea))
#

# ... ARA
timea = time.time()

opt_utils_v1.borisPush_bc_1(_particles, -dt/2, Bp, Ep, qe, me, Lz)

timeb = time.time()
print('time for new particle push: ' + str(timeb - timea))

assert(np.allclose(particles, _particles))
# ...



# ... compute intial hot electron current densities (in weak formulation)
timea = time.time()

jh = utils.hotCurrent(particles[:, 1:3], particles[:, 0], particles[:, 4], zj, bsp, qe, c)

timeb = time.time()
print('time for initial hot current computation: ' + str(timeb - timea))
# ...

# ... ARA
timea = time.time()

_jh = np.zeros(2*Nb)

opt_utils_v1.hotCurrent_bc_1(particles[:, 1:3], particles[:, 0], particles[:, 4],
                             knots, p, Nz, qe, c, _jh)

timeb = time.time()
print('time for new hot current computation: ' + str(timeb - timea))
assert(np.allclose(jh, _jh))
# ...


# ------------------------------------------------------------------------------------------------------
# ---------------------------------!to be accelerated!----------------------------------------------------

