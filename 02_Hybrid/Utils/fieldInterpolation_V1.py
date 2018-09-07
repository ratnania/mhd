# This function computes the fields at the particle positions

import numpy as np

def fieldInterpolation(particles_pos,nodes,basis,uj,p):
    
    Nb = len(nodes) - 1
    
    Ep = np.zeros((len(particles_pos),2))
    Bp = np.zeros((len(particles_pos),2))
    
    Zbin = np.digitize(particles_pos,nodes) - 1

    ex = uj[0::6]
    ey = uj[1::6]
    bx = uj[2::6]
    by = uj[3::6]
   
     
    for ie in range(0,Nb):
        
       
        indices = np.where(Zbin == ie)[0]
              
        for il in range(0,p+1):
            
            i = il + ie
            bi = basis(particles_pos[indices],i)
            
            Ep[indices,0] += ex[i%Nb]*bi
            Ep[indices,1] += ey[i%Nb]*bi
            Bp[indices,0] += bx[i%Nb]*bi
            Bp[indices,1] += by[i%Nb]*bi
       
    return Ep,Bp
