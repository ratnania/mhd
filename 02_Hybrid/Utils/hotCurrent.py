# This function computes the x and y- components of the hot current density in terms of the weak formulation

import numpy as np

def hotCurrent(particles_vel,particles_pos,particles_wk,nodes,basis,p):
    
    Nb = len(nodes) - 1
    
    jhx = np.zeros(Nb)
    jhy = np.zeros(Nb)
    
    Zbin = np.digitize(particles_pos,nodes) - 1
    
    for ie in range(0,Nb):
        
        indices = np.where(Zbin == ie)[0]
        wk = particles_wk[indices]
        vx = particles_vel[indices,0]
        vy = particles_vel[indices,1]
        
        for il in range(0,p+1):
            
            i = il + ie
            bi = basis(particles_pos[indices],i)         
         
            jhx[i%Nb] += np.einsum('i,i,i',vx,wk,bi)
            jhy[i%Nb] += np.einsum('i,i,i',vy,wk,bi)
                      

    return jhx,jhy
