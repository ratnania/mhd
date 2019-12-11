import numpy    as np
import bsplines as bsp

import time

import HyCho_PIC

from pyccel.epyccel import epyccel
HyCho_PIC_fast = epyccel(HyCho_PIC, accelerator='openmp')
print('Pyccelization done!')


# ... parameters
Nel  = 1800
p    = 3
Lz   = 327.7
el_b = np.linspace(0., Lz, Nel + 1)
bc   = False

Nbase0 = Nel + p - bc*p
Nbase1 = Nbase0 - 1 + bc

delta = Lz/Nel

T0 = bsp.make_knots(el_b, p, bc) 
T1 = T0[1:-1]

Np  = int(5e6)

xi = 0.862
dt = 0.02

pp_0 = np.asfortranarray([[1/6, -1/(2*delta), 1/(2*delta**2), -1/(6*delta**3)], [2/3, 0., -1/delta**2, 1/(2*delta**3)], [1/6, 1/(2*delta), 1/(2*delta**2), -1/(2*delta**3)], [0., 0., 0., 1/(6*delta**3)]])
pp_1 = np.asfortranarray([[1/2, -1/delta, 1/(2*delta**2)], [1/2, 1/delta, -1/delta**2], [0., 0., 1/(2*delta**2)]])

wpar = 0.2
wperp = 0.53
# ...


particles = np.zeros((Np, 5), order='F')
particles[:, 0] = np.random.rand(Np)*Lz
particles[:, 1] = np.random.randn(Np)*wperp
particles[:, 2] = np.random.randn(Np)*wperp
particles[:, 3] = np.random.randn(Np)*wpar
particles[:, 4] = np.random.rand(Np)


ex = np.zeros(Nbase0, dtype=float)
ey = np.zeros(Nbase0, dtype=float)
bx = np.zeros(Nbase1, dtype=float)
by = np.zeros(Nbase1, dtype=float)

spans0 = np.floor(particles[:, 0]*Nel/Lz).astype(int) + p

jh_x = np.empty(Nbase0, dtype=float)
jh_y = np.empty(Nbase0, dtype=float)

a = time.time()
HyCho_PIC_fast.current(particles, T0, p, spans0, jh_x, jh_y, Nbase0)
b = time.time()
print('time for hot current computation', b - a)

a = time.time()
HyCho_PIC_fast.pusher_reflecting(particles, dt, T0, T1, p, spans0, Lz, delta, ex, ey, bx, by, pp_0, pp_1, xi)
b = time.time()
print('time for pushing particles', b - a)