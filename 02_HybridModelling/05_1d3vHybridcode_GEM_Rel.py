import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import psydac.core.interface as inter

import time

import utilitis_opt as utils_opt
import utilitis_pic_Rel


#====================================================================================
#  calling epyccel
#====================================================================================
from pyccel.epyccel import epyccel
utils_pic_fast = epyccel(utilitis_pic_Rel, accelerator='openmp')
print('pyccelization of pic functions done!')
#====================================================================================



#===== saving data? (save = 1: yes, save = 0: no). If yes, name directory ===========
save = 1
title = 'test_rel.txt' 
#====================================================================================



#===== save only every saving_step-th time step =====================================
saving_step = 1
#====================================================================================



#===== physical parameters ==========================================================
eps0 = 1.0                         # vacuum permittivity
mu0 = 1.0                          # vacuum permeability
c = 1.0                            # speed of light
qe = -1.0                          # electron charge
me = 1.0                           # electron mass
B0z = 1.0                          # minimum of background magnetic field in z-direction
wce = qe*B0z/me                    # electron cyclotron frequency
wpe = 5*np.abs(wce)                # cold electron plasma frequency
nuh = 6e-3                         # ratio of cold/hot electron densities (nh/nc)
nh = nuh*wpe**2                    # hot electron density
wpar = 0.2*c                       # parallel thermal velocity of energetic particles
wperp = 0.53*c                     # perpendicular thermal velocity of energetic particles
#====================================================================================



#===== initial conditions ===========================================================
k = 2.                             # wavenumber of initial wave field perturbations
amp = 1e-4                         # amplitude of initial wave field perturbations
eps = 0.                           # amplitude of spatial pertubation of initial distribution function 


Ex0 = lambda z : 0*z               # initial Ex
Ey0 = lambda z : 0*z               # initial Ey
Bx0 = lambda z : 0*z               # initial Bx
By0 = lambda z : 0*z               # initial By
jx0 = lambda z : 0*z               # initial jcx
jy0 = lambda z : 0*z               # initial jcy
#====================================================================================



#===== numerical parameters =========================================================
Lz = 327.7                         # length of z-domain
Nel = 2200                         # number of elements z-direction
T = 1000.                          # simulation time
dt = 0.04                          # time step
p = 3                              # degree of B-spline basis functions in V0
Np = np.int(5e6)                   # number of markers
control = 1                        # control variate for noise reduction? (1: yes, 0: no)
time_integr = 1                    # do time integration? (1 : yes, 0: no)
#====================================================================================








#===== create parameter list ========================================================
pa = np.zeros(1*Nel + 5)

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
pa[13] = amp 
pa[14] = eps 
pa[15] = Lz 
pa[16] = Nel 
pa[17] = T 
pa[18] = dt 
pa[19] = p 
pa[20] = Np
pa[21] = control
pa[22] = saving_step
#====================================================================================



#===== discretization of spatial domain =============================================
dz   = Lz/Nel                                # element size
el_b = np.linspace(0, Lz, Nel + 1)           # element boundaries

Nbase   = Nel + p                            # total number of basis function
Nbase_0 = Nel                                # number of degrees of freedom
#====================================================================================



#===== some diagnostic values =======================================================
Eh_eq = Lz*nh*me/2*(wpar**2 + 2*wperp**2)    # equilibrium energetic electron energy

en_E = np.array([])                          # electric field energy
en_B = np.array([])                          # magnetic field energy
en_C = np.array([])                          # cold plasma energy
en_H = np.array([])                          # energetic electron energy
#====================================================================================



#===== initial energetic electron distribution function =============================
fh0 = lambda z, vx, vy, vz : (1 + eps*np.cos(k*z))*nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
#====================================================================================


#===== Maxwellian for control variate ===============================================
maxwell = lambda vx, vy, vz : nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
#====================================================================================


#===== sampling distribution for initial markers ====================================
g_sampling = lambda vx, vy, vz : 1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))*1/Lz
#====================================================================================



#===== spline knot vector, global mass matrices (in V0 and V1) and gradient matrix ==
Tz = inter.make_periodic_knots(p, Nbase)*Lz
tz = Tz[1:-1]

M0, C0 = utils_opt.matrixAssembly_V0(p, Nbase, Tz, True)
M1 = utils_opt.matrixAssembly_V1(p, Nbase, Tz, True)

G = utils_opt.GRAD_1d(p, Nbase, True)

print('matrix assembly done!')
#====================================================================================



#===== reserve memory for unknowns ==================================================
ex = np.empty(Nbase_0)
ey = np.empty(Nbase_0)
bx = np.empty(Nbase_0)
by = np.empty(Nbase_0)
yx = np.empty(Nbase_0)
yy = np.empty(Nbase_0)

z_old = np.empty(Np)
#====================================================================================



#===== initial coefficients with commuting projectors ===============================
ex[:] = utils_opt.PI_0_1d(Ex0, p, Nbase, Tz, True)
ey[:] = utils_opt.PI_0_1d(Ey0, p, Nbase, Tz, True)
bx[:] = utils_opt.PI_1_1d(Bx0, p, Nbase, Tz, True)
by[:] = utils_opt.PI_1_1d(By0, p, Nbase, Tz, True)
yx[:] = utils_opt.PI_0_1d(jx0, p, Nbase, Tz, True)
yy[:] = utils_opt.PI_0_1d(jy0, p, Nbase, Tz, True)

uj = np.concatenate((ex, ey, bx, by, yx, yy))

print('projection of initial fields done!')
#====================================================================================



#===== construct block matrices for field update ====================================
ZERO = np.zeros((Nbase_0, Nbase_0))

A1 = np.diag(np.ones(6*Nbase_0))
A1[0:Nbase_0, 0:Nbase_0] = M0
A1[Nbase_0:2*Nbase_0, Nbase_0:2*Nbase_0] = M0

A2 = np.block([[ZERO, ZERO, ZERO, c**2*np.dot(G.T, M1), -mu0*c**2*M0, ZERO], [ZERO, ZERO, -c**2*np.dot(G.T, M1), ZERO, ZERO, -mu0*c**2*M0], [ZERO, G, ZERO, ZERO, ZERO, ZERO], [-G, ZERO, ZERO, ZERO, ZERO, ZERO], [eps0*wpe**2*np.identity(Nbase_0), ZERO, ZERO, ZERO, ZERO, wce*np.identity(Nbase_0)], [ZERO, eps0*wpe**2*np.identity(Nbase_0), ZERO, ZERO, -wce*np.identity(Nbase_0), ZERO]])

LHS = sc.sparse.csc_matrix(A1 - 1/2*dt*A2)
RHS = sc.sparse.csc_matrix(A1 + 1/2*dt*A2)

LU = sc.sparse.linalg.splu(LHS)

print('LU factorization done!')
#====================================================================================



#===== create particles (z,vx,vy,vz,wk) and sample according to sampling distribution
particles = np.zeros((Np, 5), order='F')
particles[:, 0] = np.random.rand(Np)*Lz
particles[:, 1] = np.random.randn(Np)*wperp
particles[:, 2] = np.random.randn(Np)*wperp
particles[:, 3] = np.random.randn(Np)*wpar

jh = np.zeros(2*Nbase_0)
Fh = np.zeros(6*Nbase_0)
#====================================================================================



#===== parameters for control variate ===============================================
g0 = g_sampling(particles[:, 1], particles[:, 2], particles[:, 3])
w0 = fh0(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3])/g_sampling(particles[:, 1], particles[:, 2], particles[:, 3])
#====================================================================================



#===== initialize velocities by pushing back by -dt/2 and compute weights ===========
timea = time.time()

z_old[:] = particles[:, 0]

utils_pic_fast.borisGemRel_bc_1(particles, -dt/2, qe, me, Lz, Tz, tz, p, Nbase_0, ex, ey, bx, by, B0z, c)

particles[:, 0] = z_old
particles[:, 4] = w0 - control*maxwell(particles[:, 1], particles[:, 2], particles[:, 3])/g0

timeb = time.time()
print('time for particle push: ' + str(timeb - timea))
#====================================================================================



#===== test timing for hot current computation ======================================
timea = time.time()

utils_pic_fast.hotCurrentRel_bc_1(particles[:, 0], particles[:, 1:], Tz, p, qe, jh, c, Nbase_0)

timeb = time.time()
print('time for hot current computation: ' + str(timeb - timea))
#====================================================================================



#===== test timing for linear solver ================================================
timea = time.time()

LU.solve(RHS.dot(uj) + dt*Fh)

timeb = time.time()
print('time for solving linear system: ' + str(timeb - timea))
#====================================================================================



#===== time integration by a time step dt =========================================== 
def update():
    
    
    # ... save old positions
    z_old[:] = particles[:, 0]
    # ...
    
    
    # ... update particle velocities from n-1/2 to n+1/2 with fields at time n and positions from n to n+1 with velocities at n+1/2
    utils_pic_fast.borisGemRel_bc_1(particles, dt, qe, me, Lz, Tz, tz, p, Nbase_0, ex, ey, bx, by, B0z, c)
    # ...
    
    
    # ... update weights with control variate
    particles[:, 4] = w0 - control*maxwell(particles[:, 1], particles[:, 2], particles[:, 3])/g0
    # ...
    
    
    # ... compute hot electron current densities
    utils_pic_fast.hotCurrentRel_bc_1(1/2*(z_old + particles[:, 0]), particles[:, 1:], Tz, p, qe, jh, c, Nbase_0)
    # ...
     
    
    # ... assemble right-hand side of weak formulation
    Fh[:Nbase_0] = -c**2*mu0*jh[0::2]
    Fh[Nbase_0:2*Nbase_0] = -c**2*mu0*jh[1::2]
    # ...
    
    
    # ... time integration of E, B, jc from n to n+1 with Crank-Nicolson method (use hot current density at n+1/2) 
    uj[:] = LU.solve(RHS.dot(uj) + dt*Fh)
    
    ex[:] = uj[:Nbase_0]
    ey[:] = uj[Nbase_0:2*Nbase_0]
    bx[:] = uj[2*Nbase_0:3*Nbase_0]
    by[:] = uj[3*Nbase_0:4*Nbase_0]
    yx[:] = uj[4*Nbase_0:5*Nbase_0]
    yy[:] = uj[5*Nbase_0:]
    # ...
#====================================================================================





#===== create data file and save parameters (first row), initial fields and energies (second row)
if save == 1:
    file = open(title, 'ab')
    np.savetxt(file, np.reshape(pa, (1, 1*Nbase_0 + 5)), fmt = '%1.10e')


en_E = np.append(en_E, eps0/2*(np.dot(ex, np.dot(M0, ex)) + np.dot(ey, np.dot(M0, ey))))
en_B = np.append(en_B, eps0/(2*mu0)*(np.dot(bx, np.dot(M1, bx)) + np.dot(by, np.dot(M1, by))))
en_C = np.append(en_C, 1/(2*eps0*wpe**2)*(np.dot(yx, np.dot(M0, yx)) + np.dot(yy, np.dot(M0, yy))))
en_H = np.append(en_H, me/(2*Np)*np.dot(particles[:, 4], particles[:, 1]**2 + particles[:, 2]**2 + particles[:, 3]**2) + control*Eh_eq)

bx_save = np.append(np.copy(bx), bx[0])


if save == 1:
    data = np.append(bx, np.array([en_E[-1], en_B[-1], en_C[-1], en_H[-1], 0.]))
    np.savetxt(file, np.reshape(data, (1, 1*Nbase_0 + 5)), fmt = '%1.10e')
#====================================================================================



#===== time integration =============================================================
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

            update()

            if time_step%saving_step == 0:

                # ... add data to file
                en_E = np.append(en_E, eps0/2*(np.dot(ex, np.dot(M0, ex)) + np.dot(ey, np.dot(M0, ey))))
                en_B = np.append(en_B, eps0/(2*mu0)*(np.dot(bx, np.dot(M1, bx)) + np.dot(by, np.dot(M1, by))))
                en_C = np.append(en_C, 1/(2*eps0*wpe**2)*(np.dot(yx, np.dot(M0, yx)) + np.dot(yy, np.dot(M0, yy))))
                en_H = np.append(en_H, me/(2*Np)*np.dot(particles[:, 4], particles[:, 1]**2 + particles[:, 2]**2 + particles[:, 3]**2) + control*Eh_eq)


                bx_save = np.vstack((bx_save, np.append(np.copy(bx), bx[0])))
                
                if save == 1:
                    data = np.append(bx, np.array([en_E[-1], en_B[-1], en_C[-1], en_H[-1], (time_step + 1)*dt]))
                    np.savetxt(file, np.reshape(data, (1, 1*Nbase_0 + 5)), fmt = '%1.10e')
                # ...

            time_step += 1
        except KeyboardInterrupt:
            print('Pausing...  (Hit ENTER to continue, type quit to exit.)')
            
            if save == 1:
                file.close()
            
            try:
                response = input()
                if response == 'quit':
                    break
                print('Resuming...')
                
                if save == 1:
                    file = open(title, 'ab')
                
            except KeyboardInterrupt:
                print('Resuming...')
                
                if save == 1:
                    file = open(title, 'ab')
                
                continue
                
    if save == 1:
        file.close()
#====================================================================================


# In[33]:


tn = np.linspace(0., T, len(en_B))
plt.semilogy(tn, en_B)

omega = utils_opt.solveDispersionHybrid(k, +1, c, wce, wpe, wpar, wperp, nuh, 0.5 + 0j, 1e-8)[0]
omega_r = np.real(omega)
gamma = np.imag(omega)

t2 = np.linspace(0, 150, 100) 
plt.semilogy(t2, 1.8e-9*np.exp(gamma*2*t2), 'k--', label = 'expected growth')

