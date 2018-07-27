# Electron hybrid code


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from Utils.bsplines import Bspline
from Utils.borisPush import borisPush
from Utils.fieldInterpolation import fieldInterpolation
from Utils.hotCurrent import hotCurrent
from Utils.initialConditions import IC
from Utils.dispersionSolver import solveDispersion
import time




# ... physical constants
eps0 = 1.0                           # ... vacuum permittivity
mu0 = 1.0                            # ... vacuum permeability
c = 1.0                              # ... speed of light
qe = -1.0                            # ... electron charge
me = 1.0                             # ... electron mass
B0z = 1.0                            # ... background magnetic field in z-direction
wce = qe*B0z/me                      # ... electron cyclotron frequency
wpe = 2*np.abs(wce)                  # ... cold electron plasma frequency (normalized to electron cyclotron frequency)
nuh = 6e-2                           # ... ratio of cold/hot electron densities (nh/nc)
nh = nuh*wpe**2                      # ... hot electron density
wpar = 0.2                           # ... parallel thermal velocity of energetic particles (normalized to speed of light)
wperp = 0.63                         # ... perpendicular thermal velocity of energetic particles (normalized to speed of light)
# ...




# ... parameters for initial conditions
k = 2                                # ... wavenumber of initial wave fields
ini = 3                              # ... initial conditions for wave fields
amp = 1e-8                           # ... amplitude of initial wave fields (should be small to reduce nonlinearities)
eps = 0.0                            # ... amplitude of spatial pertubation of distribution function (1 + eps*cos(k*z))
# ...


# ... numerical parameters
Lz = 2*np.pi/k                       # ... length of z-domain
Nz = 200                             # ... number of elements z-direction
T = 80.0                             # ... simulation time
dt = 0.05                            # ... time step
p = 3                                # ... degree of B-spline basis
Lv = 8                               # ... length of v-domain (vx,vy,vz)
Nv = 76                              # ... number of cells in each v-direction (vx,vy,vz)
Np = np.int(5e4)                     # ... number of energetic macroparticles 
# ...







# ... discretization parameters
dz = Lz/Nz
zj = np.linspace(0,Lz,Nz+1)

Nt = np.int(T/dt)
tn = np.linspace(0,T,Nt+1)

dv = Lv/Nv
vj = np.linspace(-Lv/2,Lv/2,Nv+1)
# ...






# ... system matrices for fluid electrons and electromagnetic fields
A10 = np.array([0,0,0,c**2,0,0])
A11 = np.array([0,0,-c**2,0,0,0])
A12 = np.array([0,-1,0,0,0,0])
A13 = np.array([1,0,0,0,0,0])
A14 = np.array([0,0,0,0,0,0])
A15 = np.array([0,0,0,0,0,0])
A1 = np.array([A10,A11,A12,A13,A14,A15])

A20 = np.array([0,0,0,0,mu0*c**2,0])
A21 = np.array([0,0,0,0,0,mu0*c**2])
A22 = np.array([0,0,0,0,0,0])
A23 = np.array([0,0,0,0,0,0])
A24 = np.array([-eps0*wpe**2,0,0,0,0,-wce])
A25 = np.array([0,-eps0*wpe**2,0,0,wce,0])
A2 = np.array([A20,A21,A22,A23,A24,A25])

s = int(np.sqrt(A1.size))
# ...





# ... initial energetic particle distribution function (perturbed anisotropic Maxwellian)
def fh0(z,vx,vy,vz,eps):
    return nh*(1 + eps*np.cos(k*z))*1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
# ...


# ... Maxwellian for control variate
def Maxwell(vx,vy,vz):
    return nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
# ...


# ... sampling distribution for initial markers
def g_sampling(vx,vy,vz):
    return 1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))*1/Lz
# ...






# ... create B-spline basis and quadrature grid
left = np.linspace(-p*dz,-dz,p)
right = np.linspace(Lz+dz,Lz+p*dz,p)
Tbsp = np.array(list(left)  + list(zj) + list(right))
bsp = Bspline(Tbsp,p)
N = len(Tbsp) - p - 1
xi,wi = np.polynomial.legendre.leggauss(p+1)
quad_points = np.zeros((p+1)*Nz)
weights = np.zeros((p+1)*Nz)

for i in range(0,Nz):
    a1 = zj[i]
    a2 = zj[i+1]
    xis = (a2-a1)/2*xi + (a1+a2)/2
    quad_points[(p+1)*i:(p+1)*i+(p+1)] = xis
    wis = (a2-a1)/2*wi
    weights[(p+1)*i:(p+1)*i+(p+1)] = wis
# ... 






# ... matrices for linear system
Nb = N - p                        # ... number of unique B-splines for periodic boundary conditions
uhj = np.zeros((Nt+1,s*Nb))       # ... coefficients for Galerkin approximation
    
M = np.zeros((Nb,Nb))             # ... mass matrix
Mblock = np.zeros((s*Nb,s*Nb))    # ... block mass matrix
    
C = np.zeros((Nb,Nb))             # ... convection matrix
Cblock = np.zeros((s*Nb,s*Nb))    # ... block convection matrix
    
b0 = np.zeros((Nb,s))             # ... L2-projection of initial conditions

A1block = block_diag(*([A1]*Nb))  # ... block system matrix A1
A2block = block_diag(*([A2]*Nb))  # ... block system matrix A2
# ...


 



# ... assemble mass and convection matrices
timea = time.time()

for ie in range(0,Nz):
    for il in range(0,p+1):
        for jl in range(0,p+1):
                
            i = il + ie  
            j = jl + ie
                
            value_m = 0.0
            value_c = 0.0
                
            for g in range(0,p+1):
                gl = ie*(p+1) + g
                    
                value_m += weights[gl]*bsp(quad_points[gl],i,0)*bsp(quad_points[gl],j,0)
                value_c += weights[gl]*bsp(quad_points[gl],i,0)*bsp(quad_points[gl],j,1)
                    
            M[i%Nb,j%Nb] += value_m
            C[i%Nb,j%Nb] += value_c

timeb = time.time()
print('time for matrix assembly: ' + str(timeb - timea))
# ...




# ... assemble b0
timea = time.time()

for qu in range (0,s):
    for ie in range(0,Nz):
        for il in range(0,p+1):
            
            i = il + ie
            
            value_b = 0.0
            
            for g in range(0,p+1):
                gl = ie*(p+1) + g
                
                value_b += weights[gl]*IC(quad_points[gl],ini,amp,k,omega = 0)[qu]*bsp(quad_points[gl],i,0)
                
            b0[i%Nb,qu] += value_b

timeb = time.time()
print('time for vector assembly: ' + str(timeb - timea))
# ...





# ... L2-projection of initial conditions
uini = np.dot(np.linalg.inv(M),b0)
uhj[0] = np.reshape(uini,s*Nb)
# ...






# ... construct block mass and convection matrices
for i in range(0,Nb):
    for j in range(0,Nb):
        l = s*i
        m = s*j
        Mblock[l:l+s,m:m+s] = np.identity(s)*M[i,j]
        Cblock[l:l+s,m:m+s] = np.identity(s)*C[i,j]
# ...

                    

                                   
                                   
# ... create particles (z,vx,vy,vz,wk) and sample positions and velocities according to sampling distribution
particles = np.zeros((Nt+1,Np,5))
particles[0,:,0] = np.random.rand(Np)*Lz
particles[0,:,1] = np.random.randn(Np)*wperp
particles[0,:,2] = np.random.randn(Np)*wperp
particles[0,:,3] = np.random.randn(Np)*wpar
# ... 




# ... parameters for control variate
g0 = g_sampling(particles[0,:,1],particles[0,:,2],particles[0,:,3])
w0 = fh0(particles[0,:,0],particles[0,:,1],particles[0,:,2],particles[0,:,3],eps)/g_sampling(particles[0,:,1],particles[0,:,2],particles[0,:,3])
# ...






# ... initial fields at particle positions
Ep = np.zeros((Nt+1,Np,3))
Bp = np.zeros((Nt+1,Np,3))
Bp[:,:,2] = B0z

timea = time.time()

Ep[0,:,0:2],Bp[0,:,0:2] = fieldInterpolation(particles[0,:,0],zj,bsp,np.reshape(uhj[0],(Nb,s)),p)

timeb = time.time()
print('time for intial field interpolation: ' + str(timeb - timea))
# ...





# ... initialize velocities by pushing back by -dt/2 and compute weights
timea = time.time()

particles[0,:,1:4] = borisPush(particles[0],dt,Bp[0],Ep[0],qe,me,Lz)[1]
particles[0,:,4] = w0 - Maxwell(particles[0,:,1],particles[0,:,2],particles[0,:,3])/g0

timeb = time.time()
print('time for intial particle push: ' + str(timeb - timea))
#




# ... data structures for hot electron current densities
Fh = np.zeros((Nt+1,Nb*s))
jhx = np.zeros((Nt+1,Nb))
jhy = np.zeros((Nt+1,Nb))
# ...






# ... compute matrices for field update
timea = time.time()

LHS = Mblock + 1/2*dt*np.dot(Cblock,A1block) + 1/2*dt*np.dot(Mblock,A2block)
RHS = Mblock - 1/2*dt*np.dot(Cblock,A1block) - 1/2*dt*np.dot(Mblock,A2block)
LHSinv = np.linalg.inv(LHS)

timeb = time.time()
print('time for update matrix computation: ' + str(timeb - timea))
# ...





# ... time integration
for n in range(0,Nt):

    if n%50 == 0:
        print('time steps finished: ' + str(n))
    
    
    
    # ... update particle velocities
    particles[n+1,:,0],particles[n+1,:,1:4] = borisPush(particles[n],dt,Bp[n],Ep[n],qe,me,Lz)
    # ...
    
    
    
    # ... update weights with control variate
    particles[n+1,:,4] = w0 - Maxwell(particles[n+1,:,1],particles[n+1,:,2],particles[n+1,:,3])/g0
    # ...
    
    
    
    # ... compute hot electron current densities
    jhx[n],jhy[n] = hotCurrent(particles[n+1,:,1:3],1/2*(particles[n+1,:,0] + particles[n,:,0]),particles[n+1,:,4],zj,bsp,p,qe)
    # ...
    
    
    
    # ... assemble right-hand side of weak formulation
    for i in range(0,Nb):
        il = s*i
        Fh[n,il+0] = -c**2*mu0*jhx[n,i]
        Fh[n,il+1] = -c**2*mu0*jhy[n,i]
    # ...
    
    
    
    # ... time integration of E,B,jc from n to n+1 with Crank-Nicolson method (use hot current density at n+1/2) 
    uhj[n+1] = np.dot(LHSinv,np.dot(RHS,uhj[n]) + dt*Fh[n])
    uhj = np.reshape(uhj,(Nt+1,Nb,s))
    # ...
    
    
    
    # ... compute fields at particle positions with new fields 
    Ep[n+1,:,0:2],Bp[n+1,:,0:2] = fieldInterpolation(particles[n+1,:,0],zj,bsp,uhj[n+1],p)
    # ...
    
    
    # ... reshape matrix 
    uhj = np.reshape(uhj,(Nt+1,Nb*s))
    # ...
    
    
# ...


params = np.zeros(Nt+1)

params[0] = eps0
params[1] = mu0
params[2] = c
params[3] = qe 
params[4] = me 
params[5] = B0z 
params[6] = wce 
params[7] = wpe 
params[8] = nuh 
params[9] = nh 
params[10] = wpar 
params[11] = wperp 
params[12] = k 
params[13] = ini 
params[14] = amp 
params[15] = eps 
params[16] = Lz 
params[17] = Nz 
params[18] = T 
params[19] = dt 
params[20] = p 
params[21] = Lv 
params[22] = Nv 
params[23] = Np  

result = np.c_[uhj,jhx,jhy,tn,params]
np.savetxt('simulation_data.txt',result,fmt = '%1.6e')
