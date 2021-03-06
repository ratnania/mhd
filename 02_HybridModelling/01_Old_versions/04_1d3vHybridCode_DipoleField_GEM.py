
# coding: utf-8

# # 1d3v electron hybrid code for R/L-waves with stationary ions and dipole background field
# 
# ## 1. The model
# The electron hybrid model for cold electrons with linearized fluid equations and hot kinetic electrons (Vlasov equation) reads
# 
# \begin{align}
# &\frac{\partial\tilde{\mathbf{j}}_\mathrm{c}}{\partial t}=\epsilon_0\Omega_\mathrm{pe}^2\tilde{\mathbf{E}}+\tilde{\mathbf{j}}_\mathrm{c}\times\mathbf{\Omega}_\mathrm{ce}(\mathbf{x}),\\
# &\frac{\partial f_\mathrm{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\mathrm{h}+\frac{q_\mathrm{e}}{m_\mathrm{e}}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\nabla_\mathbf{v}f_\mathrm{h}=0,\\
# &\frac{\partial \tilde{\mathbf{B}}}{\partial t}=-\nabla\times\tilde{\mathbf{E}},\\
# &\frac{1}{c^2}\frac{\partial \tilde{\mathbf{E}}}{\partial t}=\nabla\times\tilde{\mathbf{B}}-\mu_0\tilde{\mathbf{j}}_\mathrm{c}-\mu_0q_\mathrm{e}\int\mathbf{v}\tilde{f}_\mathrm{h}\,\mathrm{d}^3\mathbf{v},
# \end{align}
# 
# where $\mathbf{\Omega}_\mathrm{ce}(\mathbf{x})=q_\mathrm{e}\mathbf{B}_0(\mathbf{x})/m_\mathrm{e}$ is the oriented electron cyclotron frequency corresponding to the background field $\mathbf{B}_0=\mathbf{B}_0(\mathbf{x})$ and $\Omega_{\mathrm{pe}}^2=n_{\mathrm{c}0}e^2/\epsilon_0m_\mathrm{e}$ the plasma frequency of the cold electrons. Here, only wave propagation parallel to the background magnetic field $\mathbf{B}_0=\mathbf{B}_0(\mathbf{x})$ is considered, i.e. $\textbf{k}=k\textbf{e}_z$. Therefore the nabla operator is simply $\nabla=\textbf{e}_z\partial_z$. The background magnetic field is a parabolically shaped.
# 

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import psydac.core.interface as inter

import time

import utilitis_opt as utils_opt
import utilitis_pic


#====================================================================================
#  calling epyccel
#====================================================================================
from pyccel.epyccel import epyccel
utils_pic_fast = epyccel(utilitis_pic, accelerator='openmp')
print('pyccelization of pic functions done!')
#====================================================================================



#===== saving data? (save = 1: yes, save = 0: no). If yes, name directory ===========
save = 1
title = 'test_dipole.txt' 
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

xi = 8.62e-5                       # inhomogeneity factor of background magnetic field

bcs_d = 1                          # damping of wave fields at boundaries? (1: yes, 0: no)
bcs_g = 1                          # field line dependence of initial distribution function? (1: yes, 0: no)
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
Nel = 600                          # number of elements z-direction
T = 10000.                         # simulation time
dt = 0.1                           # time step
p = 3                              # degree of B-spline basis functions in V0
Np = np.int(8e6)                   # number of markers
control = 1                        # control variate for noise reduction? (1: yes, 0: no)
time_integr = 1                    # do time integration? (1 : yes, 0: no)

Ld = 0.05*Lz                       # length of damping region at each end
#====================================================================================









#====== create parameter list =======================================================
pa = np.zeros(1*(Nel + p - 1) + 5)

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

pa[23] = xi
pa[24] = Ld

pa[29] = bcs_d
pa[30] = bcs_g
#====================================================================================



#===== discretization of spatial domain =============================================
dz   = Lz/Nel                                # element size
el_b = np.linspace(0, Lz, Nel + 1)           # element boundaries

Nbase0   = Nel + p                           # total number of basis functions in V0
Nbase0_0 = Nbase0 - 2                        # number of degrees of freedom in V1

Nbase1   = Nbase0 - 1                        # total number of basis functions in V1
Nbase1_0 = Nbase1                            # number of degrees of freedom in V1
#====================================================================================



#===== some diagnostic values =======================================================
Eh_eq = Lz*nh*me/2*(wpar**2 + 2*wperp**2)    # equilibrium energetic electron energy

en_E = np.array([])                          # electric field energy
en_B = np.array([])                          # magnetic field energy
en_C = np.array([])                          # cold plasma energy
en_H = np.array([])                          # energetic electron energy
#====================================================================================



#===== background field in z-direction ==============================================
B_background_z = lambda z : B0z*(1 + xi*(z - Lz/2)**2)
#====================================================================================



#===== initial energetic electron distribution function =============================
def fh0(z, vx, vy, vz):

    xiB = 1 - B0z/B_background_z(z)
    xiz = 1 + (wperp**2/wpar**2 - 1)*xiB*bcs_g

    return (1 + eps*np.cos(k*z))*nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - xiz*(vx**2 + vy**2)/(2*wperp**2))
#====================================================================================



#===== Maxwellian for control variate ===============================================
maxwell = lambda vx, vy, vz : nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
#====================================================================================


#===== sampling distribution for initial markers ====================================
g_sampling = lambda vx, vy, vz : 1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))*1/Lz
#====================================================================================



#===== masking function to damp wave fields near boundaries =========================
def damp(z):

    if z <= Ld:
        return np.sin(np.pi*z/(2*Ld))
    elif z >= Lz - Ld:
        return np.sin(np.pi*(Lz - z)/(2*Ld))
    else:
        return 1.0
#====================================================================================



#===== spline knot vector, global mass matrices (in V0 and V1) and gradient matrix ==
Tz = inter.make_open_knots(p, Nbase0)*Lz
tz = Tz[1:-1]

M0, C0 = utils_opt.matrixAssembly_V0(p, Nbase0, Tz, False)
M1 = utils_opt.matrixAssembly_V1(p, Nbase0, Tz, False)
Mb = utils_opt.matrixAssembly_backgroundField(p, Nbase0, Tz, False, B_background_z)

G = utils_opt.GRAD_1d(p, Nbase0, False)

print('matrix assembly done!')
#====================================================================================



#===== reserve memory for unknowns ==================================================
ex = np.empty(Nbase0)
ey = np.empty(Nbase0)
bx = np.empty(Nbase1)
by = np.empty(Nbase1)
yx = np.empty(Nbase0)
yy = np.empty(Nbase0)

uj = np.empty(4*Nbase0_0 + 2*Nbase1_0)

z_old = np.empty(Np)
#====================================================================================



#===== initial coefficients with commuting projectors ===============================
ex[:] = utils_opt.PI_0_1d(Ex0, p, Nbase0, Tz, False)
ey[:] = utils_opt.PI_0_1d(Ey0, p, Nbase0, Tz, False)
bx[:] = utils_opt.PI_1_1d(Bx0, p, Nbase0, Tz, False)
by[:] = utils_opt.PI_1_1d(By0, p, Nbase0, Tz, False)
yx[:] = utils_opt.PI_0_1d(jx0, p, Nbase0, Tz, False)
yy[:] = utils_opt.PI_0_1d(jy0, p, Nbase0, Tz, False)

uj[:] = np.concatenate((ex[1:-1], ey[1:-1], bx, by, yx[1:-1], yy[1:-1]))

print('projection of initial fields done!')
#====================================================================================



#===== construct block matrices for field update ====================================
ZERO_00 = np.zeros((Nbase0_0, Nbase0_0))
ZERO_01 = np.zeros((Nbase0_0, Nbase1_0))
ZERO_11 = np.zeros((Nbase1_0, Nbase1_0))

A1 = np.diag(np.ones(4*Nbase0_0 + 2*Nbase1_0))

A1[0:Nbase0_0, 0:Nbase0_0] = M0
A1[Nbase0_0:2*Nbase0_0, Nbase0_0:2*Nbase0_0] = M0

A1[2*Nbase0_0 + 2*Nbase1_0:3*Nbase0_0 + 2*Nbase1_0, 2*Nbase0_0 + 2*Nbase1_0:3*Nbase0_0 + 2*Nbase1_0] = M0
A1[3*Nbase0_0 + 2*Nbase1_0:4*Nbase0_0 + 2*Nbase1_0, 3*Nbase0_0 + 2*Nbase1_0:4*Nbase0_0 + 2*Nbase1_0] = M0

A2 = np.block([[ZERO_00, ZERO_00, ZERO_01, c**2*np.dot(G.T, M1), -mu0*c**2*M0, ZERO_00], [ZERO_00, ZERO_00, -c**2*np.dot(G.T, M1), ZERO_01, ZERO_00, -mu0*c**2*M0], [ZERO_01.T, G, ZERO_11, ZERO_11, ZERO_01.T, ZERO_01.T], [-G, ZERO_01.T, ZERO_11, ZERO_11, ZERO_01.T, ZERO_01.T], [eps0*wpe**2*M0, ZERO_00, ZERO_01, ZERO_01, ZERO_00, qe/me*Mb], [ZERO_00, eps0*wpe**2*M0, ZERO_01, ZERO_01, -qe/me*Mb, ZERO_00]])

LHS = sc.sparse.csc_matrix(A1 - 1/2*dt*A2)
RHS = sc.sparse.csc_matrix(A1 + 1/2*dt*A2)

LU = sc.sparse.linalg.splu(LHS)

print('LU factorization done!')


if bcs_d == 1:
    grev = inter.compute_greville(p, Nbase0, Tz)
    coll = inter.collocation_matrix(p, Nbase0, Tz, grev)[1:-1, 1:-1]
    gi = np.zeros(Nbase0)

    for i in range(Nbase0):
        gi[i] = damp(grev[i])

    Gi = np.diag(gi[1:-1])

    DAMP = np.dot(np.dot(np.linalg.inv(coll), Gi), coll)
else:
    DAMP = np.identity(Nbase0_0)
    
DAMP_block = sc.linalg.block_diag(DAMP, DAMP, np.identity(Nbase1_0), np.identity(Nbase1_0), DAMP, DAMP)
    
print('damping assembly done!')
#====================================================================================



#===== create particles (z,vx,vy,vz,wk) and sample according to sampling distribution
particles = np.zeros((Np, 5), order='F')
particles[:, 0] = np.random.rand(Np)*Lz
particles[:, 1] = np.random.randn(Np)*wperp
particles[:, 2] = np.random.randn(Np)*wperp
particles[:, 3] = np.random.randn(Np)*wpar

jh = np.zeros(2*Nbase0)
Fh = np.zeros(4*Nbase0_0 + 2*Nbase1_0)
#====================================================================================



#===== parameters for control variate ===============================================
g0 = g_sampling(particles[:, 1], particles[:, 2], particles[:, 3])
w0 = fh0(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3])/g_sampling(particles[:, 1], particles[:, 2], particles[:, 3])
#====================================================================================



#===== initialize velocities by pushing back by -dt/2 and compute weights ===========
timea = time.time()

z_old[:] = particles[:, 0]

utils_pic_fast.borisGem_bc_2(particles, -dt/2, qe, me, Tz, tz, p, ex, ey, bx, by, B0z, xi, Lz)

particles[:, 0] = z_old
particles[:, 4] = w0 - control*maxwell(particles[:, 1], particles[:, 2], particles[:, 3])/g0

timeb = time.time()
print('time for particle push: ' + str(timeb - timea))
#====================================================================================



#===== test timing for hot current computation ======================================
timea = time.time()

utils_pic_fast.hotCurrent_bc_2(particles[:, 1:3], particles[:, 0], particles[:, 4], Tz, p, qe, jh)

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
    utils_pic_fast.borisGem_bc_2(particles, dt, qe, me, Tz, tz, p, ex, ey, bx, by, B0z, xi, Lz)
    # ...
    
    
    # ... update weights with control variate
    particles[:, 4] = w0 - control*maxwell(particles[:, 1], particles[:, 2], particles[:, 3])/g0
    # ...
    
    
    # ... compute hot electron current densities
    utils_pic_fast.hotCurrent_bc_2(particles[:, 1:3], 1/2*(z_old + particles[:, 0]), particles[:, 4], Tz, p, qe, jh)
    # ...
     
    
    # ... assemble right-hand side of weak formulation
    Fh[:Nbase0_0] = -c**2*mu0*jh[2:-2][0::2]
    Fh[Nbase0_0:2*Nbase0_0] = -c**2*mu0*jh[2:-2][1::2]
    # ...
    
    
    # ... time integration of E, B, jc from n to n+1 with Crank-Nicolson method (use hot current density at n+1/2) 
    uj[:] = np.dot(DAMP_block, LU.solve(RHS.dot(uj) + dt*Fh))
       
    ex[:] = np.array([0] + list(uj[:Nbase0_0]) + [0])
    ey[:] = np.array([0] + list(uj[Nbase0_0:2*Nbase0_0]) + [0])
    bx[:] = uj[2*Nbase0_0:2*Nbase0_0 + Nbase1_0]
    by[:] = uj[2*Nbase0_0 + Nbase1_0:2*Nbase0_0 + 2*Nbase1_0]
    yx[:] = np.array([0] + list(uj[2*Nbase0_0 + 2*Nbase1_0:3*Nbase0_0 + 2*Nbase1_0]) + [0])
    yy[:] = np.array([0] + list(uj[3*Nbase0_0 + 2*Nbase1_0:4*Nbase0_0 + 2*Nbase1_0]) + [0])
    # ...
#====================================================================================



#===== create data file and save parameters (first row), initial fields and energies (second row)
if save == 1:
    file = open(title, 'ab')
    np.savetxt(file, np.reshape(pa, (1, 1*Nbase1_0 + 5)), fmt = '%1.10e')


en_E = np.append(en_E, eps0/2*(np.dot(ex[1:-1], np.dot(M0, ex[1:-1])) + np.dot(ey[1:-1], np.dot(M0, ey[1:-1]))))
en_B = np.append(en_B, eps0/(2*mu0)*(np.dot(bx, np.dot(M1, bx)) + np.dot(by, np.dot(M1, by))))
en_C = np.append(en_C, 1/(2*eps0*wpe**2)*(np.dot(yx[1:-1], np.dot(M0, yx[1:-1])) + np.dot(yy[1:-1], np.dot(M0, yy[1:-1]))))
en_H = np.append(en_H, me/(2*Np)*np.dot(particles[:, 4], particles[:, 1]**2 + particles[:, 2]**2 + particles[:, 3]**2) + control*Eh_eq)

bx_save = np.copy(bx)

if save == 1:
    data = np.append(bx, np.array([en_E[-1], en_B[-1], en_C[-1], en_H[-1], 0.]))
    np.savetxt(file, np.reshape(data, (1, 1*Nbase1_0 + 5)), fmt = '%1.10e')
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
                en_E = np.append(en_E, eps0/2*(np.dot(ex[1:-1], np.dot(M0, ex[1:-1])) + np.dot(ey[1:-1], np.dot(M0, ey[1:-1]))))
                en_B = np.append(en_B, eps0/(2*mu0)*(np.dot(bx, np.dot(M1, bx)) + np.dot(by, np.dot(M1, by))))
                en_C = np.append(en_C, 1/(2*eps0*wpe**2)*(np.dot(yx[1:-1], np.dot(M0, yx[1:-1])) + np.dot(yy[1:-1], np.dot(M0, yy[1:-1]))))
                en_H = np.append(en_H, me/(2*Np)*np.dot(particles[:, 4], particles[:, 1]**2 + particles[:, 2]**2 + particles[:, 3]**2) + control*Eh_eq)

                bx_save = np.vstack((bx_save, np.copy(bx)))
                
                if save == 1:
                    data = np.append(bx, np.array([en_E[-1], en_B[-1], en_C[-1], en_H[-1], (time_step + 1)*dt]))
                    np.savetxt(file, np.reshape(data, (1, 1*Nbase1_0 + 5)), fmt = '%1.10e')
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
