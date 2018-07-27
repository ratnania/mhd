# This function pushes particles with charge qe and mass me in the electromagnetic fields E and B by a time step dt using the Boris algorithm.

import numpy as np

def borisPush(particles,dt,B,E,qe,me,Lz):
    
    qprime = dt*qe/(2*me)
    H = qprime*B
    S = 2*H/(1 + np.linalg.norm(H,axis = 1)**2)[:,None]
    u = particles[:,1:4] + qprime*E
    uprime = u + np.cross(u + np.cross(u,H),S)
    vnew = uprime + qprime*E
    znew = (particles[:,0] + dt*vnew[:,2])%Lz
       
    return znew,vnew
