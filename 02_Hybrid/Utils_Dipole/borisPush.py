# This function pushes particles with charge qe and mass me in the electromagnetic fields E and B by a time step dt using the Boris algorithm (bcs = 1: periodic b.c., bcs = 2: reflecting b.c.)

import numpy as np

def borisPush(particles,dt,B,E,qe,me,Lz,bcs = 1):
    
    if bcs == 1:
        qprime = dt*qe/(2*me)
        H = qprime*B
        S = 2*H/(1 + np.linalg.norm(H,axis = 1)**2)[:,None]
        u = particles[:,1:4] + qprime*E
        uprime = u + np.cross(u + np.cross(u,H),S)
        vnew = uprime + qprime*E
        znew = (particles[:,0] + dt*vnew[:,2])%Lz

        return znew,vnew

    elif bcs == 2:

        qprime = dt*qe/(2*me)
        H = qprime*B
        S = 2*H/(1 + np.linalg.norm(H,axis = 1)**2)[:,None]
        u = particles[:,1:4] + qprime*E
        uprime = u + np.cross(u + np.cross(u,H),S)
        vnew = uprime + qprime*E
        znew = particles[:,0] + dt*vnew[:,2]

        indices_right = np.where(znew > Lz)[0]
        indices_left = np.where(znew < 0)[0]

        
        H_right = qprime*B[indices_right]
        S_right = 2*H_right/(1 + np.linalg.norm(H_right,axis = 1)**2)[:,None]

        H_left = qprime*B[indices_left]
        S_left = 2*H_left/(1 + np.linalg.norm(H_left,axis = 1)**2)[:,None]

        u_right = -particles[indices_right,1:4] + qprime*E[indices_right]
        u_left = -particles[indices_left,1:4] + qprime*E[indices_left]

        uprime_right = u_right + np.cross(u_right + np.cross(u_right,H_right),S_right)
        uprime_left = u_left + np.cross(u_left + np.cross(u_left,H_left),S_left)

        vnew[indices_right] = uprime_right + qprime*E[indices_right]
        vnew[indices_left] = uprime_left + qprime*E[indices_left]

        znew[indices_right] = particles[indices_right,0] + dt*vnew[indices_right,2]
        znew[indices_left] = particles[indices_left,0] + dt*vnew[indices_left,2]
       
        return znew,vnew
