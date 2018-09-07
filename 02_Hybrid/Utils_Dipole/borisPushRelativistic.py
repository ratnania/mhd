# This function pushes particles with charge qe and mass me in the electromagnetic fields E and B by a time step dt using the relativisitc Boris algorithm.

import numpy as np

def borisPushRelativistic(particles,dt,B,E,qe,me,Lz,c):

    qprime = dt*qe/(2*me)
    u = particles[:,1:4] + qprime*E
    
    H = qprime*B/np.sqrt(1 + np.linalg.norm(u,axis = 1)**2/c**2)[:,None]
    S = 2*H/(1 + np.linalg.norm(H,axis = 1)**2)[:,None]
    
    uprime = u + np.cross(u + np.cross(u,H),S)

    unew = uprime + qprime*E
    vnew = unew[:,2]/np.sqrt(1 + np.linalg.norm(unew,axis = 1)**2/c**2)
    znew = (particles[:,0] + dt*vnew)%Lz

    # Reflect particles that left the simulation boundaries at z=0 or z=Lz
    #indices_right = np.where(znew > Lz)[0]
    #indices_left = np.where(znew < 0)[0]

    #unew[indices_right] = -unew[indices_right]
    #unew[indices_left] = -unew[indices_left]
       
    return znew,unew
