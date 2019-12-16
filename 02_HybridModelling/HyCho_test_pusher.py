import numpy    as np
import bsplines as bsp
import matplotlib.pyplot as plt

import time

import HyCho_PIC


from pyccel.epyccel import epyccel
HyCho_PIC_fast = epyccel(HyCho_PIC, accelerator='openmp')
print('Pyccelization done!')


# ... parameters
Nel  = 1800
p    = 3
Lz   = 40.
el_b = np.linspace(0., Lz, Nel + 1)
bc   = False

Nbase_0 = Nel + p - bc*p
Nbase_1 = Nbase_0 - 1 + bc

delta = Lz/Nel

T0 = bsp.make_knots(el_b, p, bc) 
T1 = T0[1:-1]

Np  = int(1e2)

xi = 0.862
rel = 1
T  = 150.
dt = 0.02

Nt = int(T/dt)
tn = np.linspace(0., T, Nt + 1)

pp_0 = np.asfortranarray([[1/6, -1/(2*delta), 1/(2*delta**2), -1/(6*delta**3)], [2/3, 0., -1/delta**2, 1/(2*delta**3)], [1/6, 1/(2*delta), 1/(2*delta**2), -1/(2*delta**3)], [0., 0., 0., 1/(6*delta**3)]])
pp_1 = np.asfortranarray([[1/2, -1/delta, 1/(2*delta**2)], [1/2, 1/delta, -1/delta**2], [0., 0., 1/(2*delta**2)]])
# ...




# ... unit test for relativistic boris pusher
particles = np.zeros((Np, 5), order='F')

particles[:, 0] = Lz/2 - 2.62 

gamma = 1/np.sqrt(1 - (0.117**2 + 0.0795**2))

particles[:, 1] = 0.117*gamma
particles[:, 3] = 0.0795*gamma

spans_0 = np.floor(particles[:, 0]*Nel/Lz).astype(int) + p

ex = np.zeros(Nbase_0)
ey = np.zeros(Nbase_0)
bx = np.zeros(Nbase_1)
by = np.zeros(Nbase_1)

z_old = np.copy(particles[:, 0])

a = time.time()
HyCho_PIC_fast.pusher_reflecting(particles, dt, T0, T1, p, spans_0, Lz, delta, ex, ey, bx, by, pp_0, pp_1, xi, rel)
b = time.time()
print('time for pushing particles', b - a)

particles[:, 0] = z_old

positions = np.empty(Nt + 1)
gammas = np.empty(Nt + 1)

positions[0] = particles[0, 0]
gammas[0] = gamma


for i in range(Nt):
    HyCho_PIC_fast.pusher_reflecting(particles, dt, T0, T1, p, spans_0, Lz, delta, ex, ey, bx, by, pp_0, pp_1, xi, rel)
    positions[i + 1] = particles[0, 0]
    gammas[i + 1] = np.sqrt(1 + particles[0, 1]**2 + particles[0, 2]**2 + particles[0, 3]**2)
# ...

omega = 1.*(1 + xi*(2.62)**2)
rho = -np.cross(np.array([0.117*gamma, 0., 0.0795*gamma]), np.array([0., 0., 1.]))/omega

B = np.array([-xi*rho[0]*(-2.62), -xi*rho[1]*(-2.62), (1 + xi*(2.62)**2)])

ob = 0.117*np.sqrt(xi/np.linalg.norm(B))

phi = np.arctan(0.0795/((Lz/2 - 2.62)*ob))
A = (Lz/2 - 2.62)/np.sin(phi)
# ...


plt.plot(tn, (-2.62*np.cos(ob*tn) + 0.0795/ob*np.sin(ob*tn)) + Lz/2)
plt.plot(tn[0::200], positions[0::200], 'k+')
#plt.ylim((Lz/2 - 4, Lz/2 + 6))
plt.plot(tn, gammas/gamma + 24.)
plt.show()