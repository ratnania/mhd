# This function computes the fields at the particle postions

import numpy as np
import time

def fieldInterpolation(particles_pos,nodes,basis,uj,p):
    
    Nb = len(nodes) - 1
    
    Ep = np.zeros((len(particles_pos),2))
    Bp = np.zeros((len(particles_pos),2))
    
    Zbin = np.digitize(particles_pos,nodes) - 1
   
     
    for ie in range(0,Nb):
        
       
        indices = np.where(Zbin == ie)[0]
              
        for il in range(0,p+1):
            
            i = il + ie
            bi = basis(particles_pos[indices],i)
            
            Ep[indices,0] += uj[i%Nb,0]*bi
            Ep[indices,1] += uj[i%Nb,1]*bi
            Bp[indices,0] += uj[i%Nb,2]*bi
            Bp[indices,1] += uj[i%Nb,3]*bi
       
    return Ep,Bp
