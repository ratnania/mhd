# This function computes the x and y- components of the hot current density in terms of the weak formulation

import numpy as np

def hotCurrentRelativistic(particles_vel,particles_pos,particles_wk,nodes,basis,p,qe,c):
    
    gamma = np.sqrt(1 + np.sqrt(1 + np.linalg.norm(particles_vel,axis = 1)**2/c**2))

    Nb = len(nodes) - 1
    Np = len(particles_pos)
    
    jh = np.zeros(2*Nb)
    
    Zbin = np.digitize(particles_pos,nodes) - 1
    
    for ie in range(0,Nb):
        
        indices = np.where(Zbin == ie)[0]
        wk = particles_wk[indices]
        vx = particles_vel[indices,0]/gamma[indices]
        vy = particles_vel[indices,1]/gamma[indices]
        
        for il in range(0,p+1):
            
            i = il + ie
            bi = basis(particles_pos[indices],i)         
         
            
            jh[2*(i%Nb)] += np.einsum('i,i,i',vx,wk,bi) 
            jh[2*(i%Nb)+1] += np.einsum('i,i,i',vy,wk,bi)         

    return qe*1/Np*jh
