import numpy as np
import time
from copy import deepcopy
import Utilitis_HybridCode as utils




# ... saving data? (save = 0: no, save = 1: yes). If yes, name directory
save = 1
title = 'test.txt' 
# ...

# ... save only every saving_step-th time step
saving_step = 1
# ...



# ... physical parameters
eps0 = 1.0                         # ... vacuum permittivity
mu0 = 1.0                          # ... vacuum permeability
c = 1.0                            # ... speed of light
qe = -1.0                          # ... electron charge
me = 1.0                           # ... electron mass
B0z = 1.0                          # ... background magnetic field in z-direction
wce = qe*B0z/me                    # ... electron cyclotron frequency
wpe = 2*np.abs(wce)                # ... cold electron plasma frequency
nuh = 0.002                        # ... ratio of cold/hot electron densities (nh/nc)
nh = nuh*wpe**2                    # ... hot electron density
wpar = 0.1*c                       # ... parallel thermal velocity of energetic particles
wperp = 0.1*c                      # ... perpendicular thermal velocity of energetic particles
# ...



# ... parameters for initial conditions
k = 2                              # ... wavenumber of initial wave field perturbations
ini = 6                            # ... initial conditions for wave fields (3 : sin-perturbation of Bx, 5 : all random, 6 : all zero)
amp = 1e-4                         # ... amplitude of initial wave field perturbations
eps = 0.0                          # ... amplitude of spatial pertubation of distribution function 
# ...



# ... numerical parameters
Lz = 80                            # ... length of z-domain
Nz = 256                           # ... number of elements z-direction
T = 300.0                          # ... simulation time
dt = 0.05                          # ... time step
p = 3                              # ... degree of B-spline basis
Np = np.int(5e4)                   # ... number of energetic simulation particles 
control = 1                        # ... switch on (1) or off control variate (0)
time_integr = 1                    # ... do time integration (1 : yes, 0: no)
# ...



# ... create parameter list
pa = np.zeros(8*Nz + 5)

pa[0]  = eps0
pa[1]  = mu0
pa[2]  = c
pa[3]  = qe 
pa[4]  = me 
pa[5]  = B0z 
pa[6]  = wce 
pa[7]  = wpe 
pa[8]  = nuh 
pa[9]  = nh 
pa[10] = wpar 
pa[11] = wperp 
pa[12] = k 
pa[13] = ini 
pa[14] = amp 
pa[15] = eps 
pa[16] = Lz 
pa[17] = Nz 
pa[18] = T 
pa[19] = dt 
pa[20] = p 
pa[21] = Np
pa[22] = control
pa[23] = saving_step

pa[27] = 1
pa[28] = 1
pa[29] = 0
pa[30] = 0
# ...



# ... discretization parameters
dz = Lz/Nz
zj = np.linspace(0, Lz, Nz + 1)
# ...


# ... some diagnostic values
Eh_eq = Lz*nh*me/2*(wpar**2 + 2*wperp**2)    # ... equilibrium energetic electron energy

en_B  = np.array([])                         # ... magnetic energy
en_E  = np.array([])                         # ... electric energy
en_jc = np.array([])                         # ... cold plasma energy
en_jh = np.array([])                         # ... energetic electron energy
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




# ... time integration 
def update(uj, particles, Ep, Bp ,dt):
    
    
    # ... save old positions
    zold = deepcopy(particles[:, 0])
    # ...
    
    
    # ... update particle velocities from n-1/2 to n+1/2 with fields at time n and positions from n to n+1 with velocities at n+1/2
    znew, vnew = utils.borisPush(particles, dt, Bp, Ep, qe, me, Lz)
    # ...
    
    
    # ... update weights with control variate
    wnew = w0 - control*Maxwell(vnew[:, 0], vnew[:, 1], vnew[:, 2])/g0
    # ...
    
    
    # ... compute hot electron current densities
    jhnew = utils.hotCurrent(vnew[:, 0:2], 1/2*(znew + zold), wnew, zj, bsp, qe, c)
    # ...
     
    
    # ... assemble right-hand side of weak formulation
    Fh[0::s] = -c**2*mu0*jhnew[0::2]
    Fh[1::s] = -c**2*mu0*jhnew[1::2]
    # ...
    
    
    # ... time integration of E,B,jc from n to n+1 with Crank-Nicolson method (use hot current density at n+1/2) 
    ujnew = np.dot(LHSinv,np.dot(RHS, uj) + dt*Fh)
    # ...
    
    
    # ... compute fields at particle positions with new fields 
    Epnew, Bpnew = utils.fieldInterpolation(znew, zj, bsp, ujnew)
    # ...
    
    return znew, vnew, wnew, jhnew, ujnew, Epnew, Bpnew
# ...








# ... initial energetic particle distribution function (perturbed anisotropic Maxwellian)
def fh0(z, vx, vy, vz, eps):
    return (1 + eps*np.cos(k*z))*nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
# ...


# ... Maxwellian for control variate
def Maxwell(vx, vy, vz):
    return nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
# ...


# ... sampling distribution for initial markers
def g_sampling(vx, vy, vz):
    return 1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))*1/Lz
# ...



# ... create periodic B-spline basis and quadrature grid
bsp, N, quad_points, weights = utils.createBasis(Lz, Nz, p)
# ...



# ... some vectors for linear system
Nb = N - p                        # ... number of unique B-splines for periodic boundary conditions

uj = np.zeros(s*Nb)               # ... coefficients for Galerkin approximation
Fh = np.zeros(s*Nb)               # ... RHS of matrix system

u0 = np.zeros((Nb, s))            # ... L2-projection of initial conditions
# ...





# ... assemble mass and convection matrices
timea = time.time()

M, C = utils.matrixAssembly(bsp, weights, quad_points, B_background_z, 1)[0:2]

timeb = time.time()
print('time for matrix assembly: ' + str(timeb - timea))
# ...





# ... assemble u0
timea = time.time()

for qu in range (0, s):

    def initial(z):
        return utils.IC(z, ini, amp, k, omega = 0)[qu]

    u0[:, qu] = utils.L2proj(bsp, Lz, quad_points, weights, M, initial)

uj = np.reshape(u0, s*Nb)

timeb = time.time()
print('time for initial vector assembly: ' + str(timeb - timea))
# ...




# ... construct block mass and convection matrices
timea = time.time()

M_block = np.kron(M, np.identity(6))
C_tilde = np.kron(C, A1)
M_tilde = np.kron(M, A2)

timeb = time.time()
print('time for block matrix construction: ' + str(timeb - timea))
# ...



# ... create particles (z, vx, vy, vz, wk) and sample positions and velocities according to sampling distribution
particles = np.zeros((Np, 5))
particles[:, 0] = np.random.rand(Np)*Lz
particles[:, 1] = np.random.randn(Np)*wperp
particles[:, 2] = np.random.randn(Np)*wperp
particles[:, 3] = np.random.randn(Np)*wpar
# ...




# ... parameters for control variate
g0 = g_sampling(particles[:, 1], particles[:, 2], particles[:, 3])
w0 = fh0(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3], eps)/g_sampling(particles[:, 1], particles[:, 2], particles[:, 3])
# ...



# ... initial fields at particle positions
Ep = np.zeros((Np, 3))
Bp = np.zeros((Np, 3))
Bp[:, 2] = B0z

timea = time.time()

Ep[:, 0:2], Bp[:, 0:2] = utils.fieldInterpolation(particles[:, 0], zj, bsp, uj)

timeb = time.time()
print('time for initial field interpolation: ' + str(timeb - timea))
# ...





# ... initialize velocities by pushing back by -dt/2, compute weights and energy of hot particles
timea = time.time()

particles[:, 1:4] = utils.borisPush(particles, -dt/2, Bp, Ep, qe, me, Lz)[1]
particles[:, 4] = w0 - control*Maxwell(particles[:, 1], particles[:, 2], particles[:, 3])/g0

timeb = time.time()
print('time for initial particle push: ' + str(timeb - timea))
#





# ... compute matrices for field update
timea = time.time()

LHS = M_block + 1/2*dt*C_tilde + 1/2*dt*M_tilde
RHS = M_block - 1/2*dt*C_tilde - 1/2*dt*M_tilde

LHSinv = np.linalg.inv(LHS)

timeb = time.time()
print('time for update matrix computation: ' + str(timeb - timea))
# ...





# ... create data file and save parameters (first row), initial fields and energies (second row)
if save == 1:
    file = open(title, 'ab')
    np.savetxt(file, np.reshape(pa, (1, 8*Nb + 5)), fmt = '%1.10e')
    data = np.append(uj, np.zeros(2*Nb))


en_E  = np.append(en_E, eps0/2*(np.dot(uj[0::s], np.dot(M, uj[0::s])) + np.dot(uj[1::s], np.dot(M, uj[1::s]))))
en_B  = np.append(en_B, eps0/(2*mu0)*(np.dot(uj[2::s], np.dot(M, uj[2::s])) + np.dot(uj[3::s], np.dot(M, uj[3::s]))))
en_jc = np.append(en_jc, 1/(2*eps0*wpe**2)*(np.dot(uj[4::s], np.dot(M, uj[4::s])) + np.dot(uj[5::s], np.dot(M, uj[5::s]))))
en_jh = np.append(en_jh, me/(2*Np)*np.dot(particles[:, 4], particles[:, 1]**2 + particles[:, 2]**2 + particles[:, 3]**2) + control*Eh_eq)

bx_save = np.append(deepcopy(uj[2::s]), uj[2])

if save == 1:
    data = np.append(data, np.array([en_E[-1], en_B[-1], en_jc[-1], en_jh[-1]]))
    data = np.append(data, 0.)
    np.savetxt(file, np.reshape(data, (1, 8*Nb + 5)), fmt = '%1.10e')
# ...



# ... time integration
if time_integr == 1:

    print('start time integration! (number of time steps : ' + str(int(T/dt)) + ')')
    time_step = 0

    while True:

        try:
            if time_step*dt >= T:
                if save == 1:
                    file.close()
                break

            if time_step%50 == 0:
                print('time steps finished: ' + str(time_step))

            particles[:, 0], particles[:, 1:4], particles[:, 4], jh, uj, Ep[:, 0:2], Bp[:, 0:2] = update(uj, particles, Ep, Bp, dt)

            if time_step%saving_step == 0:

                # ... add data to file
                if save == 1:
                    data = np.append(uj, jh)

                en_E  = np.append(en_E, eps0/2*(np.dot(uj[0::s], np.dot(M, uj[0::s])) + np.dot(uj[1::s], np.dot(M, uj[1::s]))))
                en_B  = np.append(en_B, eps0/(2*mu0)*(np.dot(uj[2::s], np.dot(M, uj[2::s])) + np.dot(uj[3::s], np.dot(M, uj[3::s]))))
                en_jc = np.append(en_jc, 1/(2*eps0*wpe**2)*(np.dot(uj[4::s], np.dot(M, uj[4::s])) + np.dot(uj[5::s], np.dot(M, uj[5::s]))))
                en_jh = np.append(en_jh, me/(2*Np)*np.dot(particles[:, 4], particles[:, 1]**2 + particles[:, 2]**2 + particles[:, 3]**2) + control*Eh_eq)

                bx_save = np.vstack((bx_save, np.append(deepcopy(uj[2::s]), uj[2])))

                if save == 1:
                    data = np.append(data, np.array([en_E[-1], en_B[-1], en_jc[-1], en_jh[-1]]))
                    data = np.append(data, (time_step + 1)*dt)
                    np.savetxt(file, np.reshape(data, (1, 8*Nb + 5)), fmt = '%1.10e')
                # ...

            time_step += 1
        except KeyboardInterrupt:
            print('Pausing...  (Hit ENTER to continue, type quit to exit.)')
            try:
                response = input()
                if response == 'quit':
                    break
                print('Resuming...')
            except KeyboardInterrupt:
                print('Resuming...')
                continue
                
    if save == 1:
        file.close()
# ...
