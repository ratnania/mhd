import numpy    as np
import bsplines as bsp

import time

import HyCho_PIC

from pyccel.epyccel import epyccel

pic = epyccel(HyCho_PIC, accelerator='openmp')

print('Pyccelization done!')


# ... parameters
Nel    = 1800
p      = 3
Lz     = 325.
el_b   = np.linspace(0., Lz, Nel + 1)
bc     = False

Nbase0 = Nel + p - bc*p
Nbase1 = Nbase0 - 1 + bc

delta  = Lz/Nel

T0     = bsp.make_knots(el_b, p, bc) 
T1     = T0[1:-1]

Np     = int(5e6)

xi     = 0.862
rel    = 1
dt     = 0.04

pp0    = np.asfortranarray([[1/6, -1/(2*delta), 1/(2*delta**2), -1/(6*delta**3)], [2/3, 0., -1/delta**2, 1/(2*delta**3)], [1/6, 1/(2*delta), 1/(2*delta**2), -1/(2*delta**3)], [0., 0., 0., 1/(6*delta**3)]])
pp1    = np.asfortranarray([[1/2, -1/delta, 1/(2*delta**2)], [1/2, 1/delta, -1/delta**2], [0., 0., 1/(2*delta**2)]])/delta

wpar   = 0.2
wperp  = 0.53
# ...


particles       = np.empty((Np, 5), order='F', dtype=float)
particles[:, 0] = np.random.rand(Np)*Lz
particles[:, 1] = np.random.randn(Np)*wperp
particles[:, 2] = np.random.randn(Np)*wperp
particles[:, 3] = np.random.randn(Np)*wpar
particles[:, 4] = np.random.rand(Np)
                                                                                                                      
spans0 = np.floor(particles[:, 0]*Nel/Lz).astype(int) + p

ex = np.empty(Nbase0, dtype=float)
ey = np.empty(Nbase0, dtype=float)
bx = np.empty(Nbase1, dtype=float)
by = np.empty(Nbase1, dtype=float)
                                                                                                                      
ex[:] = np.random.rand(Nbase0)
ex[:] = np.random.rand(Nbase0)
bx[:] = np.random.rand(Nbase1)
by[:] = np.random.rand(Nbase1)                    
                                                                                                                      

jh_x = np.empty(Nbase0, dtype=float)
jh_y = np.empty(Nbase0, dtype=float)
                                                                                                                 
                                                                                                                      
timea = time.time()
pic.current(particles[:, 0], particles[:, 1:], T0, p, spans0, jh_x, jh_y, Nbase0, rel)
timeb = time.time()
print('time for hot current computation', timeb - timea)


if bc == False:
    timea = time.time()
    pic.pusher_reflecting(particles, dt, T0, T1, p, spans0, Lz, delta, ex, ey, bx, by, pp0, pp1, xi, rel)
    timeb = time.time()
    print('time for pushing particles', timeb - timea)
    
elif bc == True:
    timea = time.time()
    pic.pusher_periodic(particles, dt, T0, T1, p, spans0, Lz, Nbase0, ex, ey, bx, by, pp0, pp1, rel)
    timeb = time.time()
    print('time for pushing particles', timeb - timea)