
# coding: utf-8

# # Electron hybrid code for R/L-waves with stationary ions
# 
# ## 1. The model
# The electron hybrid model for cold fluid electrons and hot kinetic electrons reads
# 
# \begin{align}
# \frac{\partial \textbf{j}_\text{c}}{\partial t} = \epsilon_0\Omega_{\text{pe}}^2\textbf{E} + \Omega_\text{ce}\textbf{j}_\text{c}\times\textbf{e}_z,  \\
# \frac{1}{c^2}\frac{\partial \textbf{E}}{\partial t} = \nabla\times\textbf{B} - \mu_0(\textbf{j}_\text{c} + \textbf{j}_\text{h}), \\
# \frac{\partial \textbf{B}}{\partial t}=-\nabla\times\textbf{E}, \\
# \frac{\partial f_\text{h}}{\partial t} + \textbf{v}\cdot\nabla f_\text{h} + \frac{q}{m}(\textbf{E}+\textbf{v}\times\textbf{B})\cdot\nabla_v f_\text{h}=0, \\
# \textbf{j}_\text{h} = q\int\text{d}^3\textbf{v}\,\textbf{v}f_\text{h}.
# \end{align}
# 
# where $\Omega_\text{ce}=\frac{q B_0}{m}$ is the signed cyclotron frequency and $\Omega_{\text{pe}}^2=\frac{n_\text{c}e^2}{\epsilon_0 m}$ the plasma frequency of the cold electrons. Here, only wave propagation parallel to the background magnetic field $\textbf{B}_0=B_0\textbf{e}_z$ is considered, i.e. $\textbf{k}=k\textbf{e}_z$. Therefore the nabla operator is simply $\nabla=\textbf{e}_z\partial_z$.
# 
# The first three equations are written in the compact form 
# 
# \begin{align}
# \partial_t \textbf{U}+A_1\partial_z \textbf{U}+A_2\textbf{U}=\textbf{F},
# \end{align}
# 
# for the electromagnetic fields $\textbf{E},\textbf{B}$ and the cold current density $\textbf{j}_\text{c}$, i.e. $\textbf{U}=(E_x,E_y,B_x,B_y,j_{\text{c}x},j_{\text{c}y})$. The z-components do not appear because they correspond to electrostatic waves which are not considered in this work. The matrices are
# 
# \begin{align}
# A_1=
# \begin{pmatrix}
# 0 &0  &0 &c^2  &0 &0 \\
# 0 &0  &-c^2 &0 &0 &0 \\
# 0 &-1  &0 &0 &0 &0  \\
# 1 &0  &0 &0 &0 &0  \\
# 0 &0  &0 &0 &0 &0   \\
# 0 &0  &0 &0 &0 &0 
# \end{pmatrix}
# \end{align}
# 
# and 
# 
# \begin{align}
# A_2=
# \begin{pmatrix}
# 0 &0 &0 &0 &\mu_0c^2 &0 \\
# 0 &0 &0 &0 &0 &\mu_0c^2 \\
# 0 &0 &0 &0 &0 &0 \\
# 0 &0 &0 &0 &0 &0 \\
# -\epsilon_0\Omega_{\text{pe}}^2 &0 &0 &0 &0 &-\Omega_{\text{ce}} \\
# 0 &-\epsilon_0\Omega_{\text{pe}}^2 &0 &0 &\Omega_{\text{ce}} &0 \\
# \end{pmatrix}
# \end{align}
# 
# with $\Omega_{\text{ce}}=-\frac{eB_0}{m}<0$ for electrons. The inhomogeneity is 
# 
# \begin{align}
# \textbf{F}=
# \begin{pmatrix}
# -\mu_0c^2 j_{\text{h}x} \\
# -\mu_0c^2 j_{\text{h}y} \\
# 0 \\
# 0 \\
# 0 \\
# 0
# \end{pmatrix}.
# \end{align}
# 
# 
# ## 2. Dispersion relation
# Linear theory of the above model leads to the following general dispersion relation for an arbitrary equilibrium distribution function $f^0=f^0(v_\parallel,v_\bot)$:
# 
# 
# \begin{align}
# D_{\text{R/L}}(k,\omega)=1-\frac{c^2k^2}{\omega^2}-\frac{\Omega_{\text{pe}}^2}{\omega(\omega\pm\Omega_{\text{ce}})}+\nu_\text{h}\frac{\Omega_{\text{pe}}^2}{\omega}\int\text{d}^3\textbf{v}\frac{v_\bot}{2}\frac{\hat{G}f_\text{h}^0}{\omega\pm\Omega_{\text{ce}}-kv_\parallel}=0.
# \end{align}
# 
# Here $\nu_\text{h}=n_\text{h}/n_\text{c}\ll1$ is the ratio between the hot and cold electron number densities, respectively, $\text{d}^3\textbf{v}=\text{d}v_\parallel\text{d}v_\bot v_\bot 2\pi$ and the differential operator
# 
# \begin{align}
# \hat{G}=\frac{\partial}{\partial v_\bot}+\frac{k}{\omega}\left(v_\bot\frac{\partial}{\partial v_\parallel}-v_\parallel\frac{\partial}{\partial v_\bot}\right).
# \end{align}
# 
# For an anisotropic Maxwellian 
# 
# \begin{align}
# f^0(v_\parallel,v_\bot) = \frac{1}{(2\pi)^{3/2}w_\parallel w_\bot^2}\exp\left(-\frac{v_\parallel^2}{2w_\parallel^2}-\frac{v_\bot^2}{2w_\bot^2}\right)
# \end{align}
# 
# the dispersion relation is given by
# 
# \begin{align}
# D_{\text{R/L}}(k,\omega)=D_{\text{cold,R/L}}(k,\omega)+\nu_\text{h}\frac{\Omega_{\text{pe}}^2}{\omega^2}\left[\frac{\omega}{k\sqrt{2}w_\parallel}Z(\xi^{\pm})-\left(1-\frac{w_\bot^2}{w_\parallel^2}\right)(1+\xi^{\pm} Z(\xi^{\pm}))\right]=0, 
# \end{align}
# 
# where $Z$ is the plasma dispersion function and 
# 
# \begin{align}
# \xi^{\pm} = \frac{\omega\pm\Omega_\text{ce}}{k\sqrt{2}w_\parallel}.
# \end{align}
# 
# ## 3. Discretization
# For the fields, B-spline Finite Elements are used together with a Crank-Nicolson time discretization which leads to the following matrix formulation:
# 
# \begin{align}
# \left[M+\frac{1}{2}\Delta tCA_1+\frac{1}{2}\Delta tMA_2\right]\textbf{U}^{n+1}=\left[M-\frac{1}{2}\Delta tCA_1-\frac{1}{2}\Delta tMA_2\right]\textbf{U}^{n} + \Delta t \tilde{\textbf{F}}^{n+1/2},
# \end{align}
# 
# with the mass and convection matrices
# 
# \begin{align}
# M_{ij}=\int\varphi_i\varphi_j\,\text{d}z, \\
# C_{ij}=\int\varphi_i\varphi_j^\prime\,\text{d}z
# \end{align}
# 
# The hot current density is obtained using PIC techniques, i.e. the distribution function reads
# 
# \begin{align}
# f_\text{h}(z,\textbf{v},t) \approx \frac{1}{N_k}\sum_k w_k\delta(z-z_k(t))\delta(\textbf{v}-\textbf{v}_k(t))
# \end{align},
# 
# with the orbit equations
# 
# \begin{align}
# \frac{\text{d}z_k}{dt}=v_{kz}, \\
# \frac{\text{d}\textbf{v}_k}{dt}=\frac{q}{m}(\textbf{E}_k+\textbf{v}_k\times\textbf{B}_k).
# \end{align}
# With the definition of $f_\text{h}$ the inhomogeneity $\tilde{\textbf{F}}^{n+1/2}$ is
# 
# \begin{align}
# \tilde{F}^{n+1/2}_i = -c^2\mu_0q\frac{1}{N_k}\sum_k w_k\textbf{v}_k^{n+1/2}\varphi_i(z_k^{n+1/2}).
# \end{align}

# In[13]:


import numpy as np
import time
from copy import deepcopy
from scipy.linalg import block_diag

from Utils.bsplines import Bspline
from Utils.borisPush import borisPush
from Utils.fieldInterpolation_V2 import fieldInterpolation
from Utils.hotCurrent import hotCurrent
from Utils.initialConditions import IC
from Utils.dispersionSolver import solveDispersionHybrid
from Utils.createBasis import createBasis
from Utils.matrixAssembly import matrixAssembly




restart = 0                        # ... start the simulation from the beginning (0) or continue (1)
title = 'Results/simulation_data_T=60.txt'   # ... directory for saving data


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
wperp = 0.63*c                     # ... perpendicular thermal velocity of energetic particles
# ...




# ... parameters for initial conditions
k = 2                              # ... wavenumber of initial wave fields
ini = 3                            # ... initial conditions for wave fields
amp = 1e-6                         # ... amplitude of initial wave fields
eps = 0.0                          # ... amplitude of spatial pertubation of distribution function 
# ...



# ... numerical parameters
Lz = 2*np.pi/k                     # ... length of z-domain
Nz = 200                           # ... number of elements z-direction
T = 60.0                           # ... simulation time
dt = 0.05                          # ... time step
p = 3                              # ... degree of B-spline basis
Lv = 8                             # ... length of v-domain in each direction (vx,vy,vz)
Nv = 76                            # ... number of cells in each v-direction (vx,vy,vz)
Np = np.int(5e4)                   # ... number of energetic simulation particles 
# ...



# ... create parameter list
pa = np.zeros(8*Nz + 1)

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
pa[21] = Lv 
pa[22] = Nv 
pa[23] = Np 
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
A10 = np.array([0,-1])
A11 = np.array([+1,0])
A1 = np.array([A10,A11])

A20 = np.array([0,+c**2])
A21 = np.array([-c**2,0])
A2 = np.array([A20,A21])

A30 = np.array([mu0*c**2,0])
A31 = np.array([0,mu0*c**2])
A3 = np.array([A30,A31])

A40 = np.array([0,-wce])
A41 = np.array([+wce,0])
A4 = np.array([A40,A41])

s = int(np.sqrt(A1.size))
# ...




# ... update E-field, B-field and jc-field
def updateFields(ej,bj,ycj,yhj,U1,U2,U3,U4,U5,U6,U7,dt):
    
    ejnew = np.dot(U1,ej) - dt*np.dot(U2,bj) - np.dot(U3,ycj) - dt*np.dot(U4,yhj)
    bjnew = bj - dt/2*np.dot(U5,ej + ejnew)
    ycjnew = dt/2*eps0*wpe**2*np.dot(U6,ej + ejnew) + np.dot(U7,ycj)
    
    return ejnew,bjnew,ycjnew
# ...





# ... time integration 
def update(ej,bj,ycj,yhj,particles,Ep,Bp,dt):
    
    
    # ... save old positions
    zold = deepcopy(particles[:,0])
    # ...
    
    
    # ... update particle velocities from n-1/2 to n+1/2 with fields at time n and positions from n to n+1 with velocities at n+1/2
    znew,vnew = borisPush(particles,dt,Bp,Ep,qe,me,Lz)
    # ...
    
    
    # ... update weights with control variate
    wnew = w0 - Maxwell(vnew[:,0],vnew[:,1],vnew[:,2])/g0
    # ...
    
    
    # ... compute hot electron current densities
    yhjnew = hotCurrent(vnew[:,0:2],1/2*(znew + zold),wnew,zj,bsp,p,qe)
    # ...
     
    
    # ... time integration of E,B,jc from n to n+1 with Crank-Nicolson method (use hot current density at n+1/2) 
    ejnew,bjnew,ycjnew = updateFields(ej,bj,ycj,yhj,U2,U3,U4,U5,U6,B1blockinv,U7,dt)
    # ...
    
    
    # ... compute fields at particle positions with new fields 
    Epnew,Bpnew = fieldInterpolation(znew,zj,bsp,ej,bj,p)
    # ...
    
    
    return znew,vnew,wnew,ejnew,bjnew,ycjnew,yhjnew,Epnew,Bpnew
# ...





if restart == 0:

    # ... initial energetic particle distribution function (perturbed anisotropic Maxwellian)
    def fh0(z,vx,vy,vz,eps):
        return (1 + eps*np.cos(k*z))*nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
    # ...


    # ... Maxwellian for control variate
    def Maxwell(vx,vy,vz):
        return nh/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))
    # ...


    # ... sampling distribution for initial markers
    def g_sampling(vx,vy,vz):
        return 1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-vz**2/(2*wpar**2) - (vx**2 + vy**2)/(2*wperp**2))*1/Lz
    # ...



    # ... create B-spline basis and quadratur grid
    bsp,N,quad_points,weights = createBasis(Lz,Nz,p)
    # ...



    # ... matrices for linear systems
    Nb = N - p                        # ... number of unique B-splines for periodic boundary conditions


    ej = np.zeros(s*Nb)               # ... coefficients for Galerkin approximation (E-field)
    bj = np.zeros(s*Nb)               # ... coefficients for Galerkin approximation (B-field)
    ycj = np.zeros(s*Nb)              # ... coefficients for Galerkin approximation (jc-field)
    yhj = np.zeros(s*Nb)              # ... weak formulation of hot current density (jh)


    Mblock = np.zeros((s*Nb,s*Nb))    # ... block mass matrix   
    Cblock = np.zeros((s*Nb,s*Nb))    # ... block convection matrix

    u0 = np.zeros((Nb,3*s))           # ... L2-projection of initial conditions

    A1block = block_diag(*([A1]*Nb))  # ... block system matrix A1
    A2block = block_diag(*([A2]*Nb))  # ... block system matrix A2
    A3block = block_diag(*([A3]*Nb))  # ... block system matrix A3
    # ...




    # ... assemble mass and convection matrices
    timea = time.time()

    M,C = matrixAssembly(bsp,p,Nz,weights,quad_points)

    timeb = time.time()
    print('time for matrix assembly: ' + str(timeb - timea))
    # ...



    # ... assemble u0
    timea = time.time()

    for qu in range (0,3*s):
        for ie in range(0,Nz):
            for il in range(0,p+1):

                i = il + ie

                value_u = 0.0

                for g in range(0,p+1):
                    gl = ie*(p+1) + g

                    value_u += weights[gl]*IC(quad_points[gl],ini,amp,k,omega = 0)[qu]*bsp(quad_points[gl],i,0)

                u0[i%Nb,qu] += value_u

    timeb = time.time()
    print('time for vector assembly: ' + str(timeb - timea))
    # ...



    # ... L2-projection of initial conditions
    uini = np.dot(np.linalg.inv(M),u0)

    ej = np.reshape(uini[:,0:2],s*Nb)
    bj = np.reshape(uini[:,2:4],s*Nb)
    ycj = np.reshape(uini[:,4:6],s*Nb)
    # ...


    # ... construct block mass and convection matrices
    for i in range(0,Nb):
        for j in range(0,Nb):
            l = s*i
            m = s*j
            Mblock[l:l+s,m:m+s] = np.identity(s)*M[i,j]
            Cblock[l:l+s,m:m+s] = np.identity(s)*C[i,j]
    # ...



    # ... create particles (z,vx,vy,vz,wk) and sample positions and velocities according to sampling distribution g
    particles = np.zeros((Np,5))
    particles[:,0] = np.random.rand(Np)*Lz
    particles[:,1] = np.random.randn(Np)*wperp
    particles[:,2] = np.random.randn(Np)*wperp
    particles[:,3] = np.random.randn(Np)*wpar
    # ...




    # ... compute parameters for control variate and initial weights
    g0 = g_sampling(particles[:,1],particles[:,2],particles[:,3])
    w0 = fh0(particles[:,0],particles[:,1],particles[:,2],particles[:,3],eps)/g_sampling(particles[:,1],particles[:,2],particles[:,3])
    particles[:,4] = w0 - Maxwell(particles[:,1],particles[:,2],particles[:,3])/g0
    # ...



    # ... initial fields at particle positions
    Ep = np.zeros((Np,3))
    Bp = np.zeros((Np,3))
    Bp[:,2] = B0z

    timea = time.time()

    Ep[:,0:2],Bp[:,0:2] = fieldInterpolation(particles[:,0],zj,bsp,ej,bj,p)

    timeb = time.time()
    print('time for intial field interpolation: ' + str(timeb - timea))
    # ...




    # ... initialize velocities by pushing back by -dt/2 and compute weights
    timea = time.time()

    particles[:,1:4] = borisPush(particles,-dt/2,Bp,Ep,qe,me,Lz)[1]
    particles[:,4] = w0 - Maxwell(particles[:,1],particles[:,2],particles[:,3])/g0

    timeb = time.time()
    print('time for intial particle push: ' + str(timeb - timea))
    #






    # ... compute needed matrices
    timea = time.time()

    Minv = np.linalg.inv(Mblock)

    B1 = np.identity(s) + dt/2*A4
    B2 = np.identity(s) - dt/2*A4

    B1block = block_diag(*([B1]*Nb)) 
    B2block = block_diag(*([B2]*Nb))
    B1blockinv = np.linalg.inv(B1block)

    U1 = Mblock - dt**2/4*np.dot(np.dot(Cblock,np.dot(A2block,Minv)),np.dot(Cblock,A1block)) + dt**2/4*np.dot(Mblock,np.dot(A3block,B1blockinv))*eps0*wpe**2
    U1inv = np.linalg.inv(U1)

    U2 = np.dot(U1inv,Mblock + dt**2/4*np.dot(np.dot(Cblock,A2block),np.dot(np.dot(Minv,Cblock),A1block)) - dt**2/4*np.dot(Mblock,np.dot(A3block,B1blockinv))*eps0*wpe**2)
    U3 = np.dot(U1inv,np.dot(Cblock,A2block))
    U4 = np.dot(U1inv,dt/2*np.dot(Mblock,A3block) + dt/2*np.dot(np.dot(Mblock,A3block),np.dot(B1blockinv,B2block)))
    U5 = np.dot(U1inv,A3block)
    U6 = np.dot(np.dot(Minv,Cblock),A1block)
    U7 = np.dot(B1blockinv,B2block)

    timeb = time.time()
    print('time for matrix computations: ' + str(timeb - timea))
    # ...




    # ... create data file and save parameters (first row) and initial fields (second row)
    file = open(title,'ab')


    np.savetxt(file,np.reshape(pa,(1,8*Nb + 1)),fmt = '%1.6e')

    data = np.append([ej,bj,ycj],yhj)
    data = np.append(data,tn[0])
    np.savetxt(file,np.reshape(data,(1,8*Nb + 1)),fmt = '%1.6e')
    # ...






    # ... time loop
    print('total number of time steps: ' + str(Nt))

    for n in range(0,Nt):

        if n%50 == 0:
            print('time steps finished: ' + str(n))

        particles[:,0],particles[:,1:4],particles[:,4],ej,bj,ycj,yhj,Ep[:,0:2],Bp[:,0:2] = update(ej,bj,ycj,yhj,particles,Ep,Bp,dt)

        # ... add data to file
        data = np.append([ej,bj,ycj],yhj)
        data = np.append(data,tn[n+1])
        np.savetxt(file,np.reshape(data,(1,8*Nb + 1)),fmt = '%1.6e')
        # ...
    # ...


    file.close()
    
    
    
    
    
if restart == 1:
    
    # ... open data file that wasn't finished
    file = open(title,'ab')
    # ...


    # ... time loop
    print('total number of time steps: ' + str(Nt))

    for i in range(n,Nt):

        if i%50 == 0:
            print('time steps finished: ' + str(i))

        particles[:,0],particles[:,1:4],particles[:,4],ej,bj,ycj,yhj,Ep[:,0:2],Bp[:,0:2] = update(ej,bj,ycj,yhj,particles,Ep,Bp,dt)

        # ... add data to file
        data = np.append([ej,bj,ycj],yhj)
        data = np.append(data,tn[i+1])
        np.savetxt(file,np.reshape(data,(1,8*Nb + 1)),fmt = '%1.6e')
        # ...
    # ...


    file.close()   

