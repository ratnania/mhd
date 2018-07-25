# This function computes the x and y- components of the hot current density in terms of the weak formulation

import numpy as np

def hotCurrent(particles_vel,particles_pos,particles_wk,nodes,basis,p):
    
    Nb = len(nodes) - 1
    
    jhx = np.zeros(Nb)
    jhy = np.zeros(Nb)
    
    Zbin = np.digitize(particles_pos,nodes) - 1
    
    for ie in range(0,Nb):
        indices = np.where(Zbin == ie)[0]
        
        for il in range(0,p+1):
            i = il + ie
            
            jhx[i%Nb] += np.einsum('i,i,i',particles_vel[indices,0],particles_wk[indices],basis(particles_pos[indices],i))
            jhy[i%Nb] += np.einsum('i,i,i',particles_vel[indices,1],particles_wk[indices],basis(particles_pos[indices],i))          

    return jhx,jhy
