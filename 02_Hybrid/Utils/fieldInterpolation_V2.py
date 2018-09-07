# This function computes the fields at the particle positions

import numpy as np

def fieldInterpolation(particles_pos,nodes,basis,ej,bj,p):
    
    Nb = len(nodes) - 1
    
    Ep = np.zeros((len(particles_pos),2))
    Bp = np.zeros((len(particles_pos),2))
    
    Zbin = np.digitize(particles_pos,nodes) - 1

    ex = ej[0::2]
    ey = ej[1::2]
    bx = bj[0::2]
    by = bj[1::2]
   
     
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
