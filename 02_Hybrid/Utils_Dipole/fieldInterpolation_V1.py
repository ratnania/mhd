# This function computes the fields at the particle positions (bcs = 1: periodic b.c., bcs = 2: Dirichlet b.c.)

import numpy as np

def fieldInterpolation(particles_pos,nodes,basis,uj,p,bcs = 1):
    

    if bcs == 1:
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

    elif bcs == 2:
        Nb = len(nodes) - 1
    
        Ep = np.zeros((len(particles_pos),2))
        Bp = np.zeros((len(particles_pos),2))
    
        Zbin = np.digitize(particles_pos,nodes) - 1

        ex = np.array([0] + list(uj[0::6]) + [0]) 
        ey = np.array([0] + list(uj[1::6]) + [0]) 
        bx = np.array([0] + list(uj[2::6]) + [0]) 
        by = np.array([0] + list(uj[3::6]) + [0]) 


        for ie in range(0,Nb):

            indices = np.where(Zbin == ie)[0]

            for il in range(0,p+1):

                i = il + ie
                bi = basis(particles_pos[indices],i)
            
                Ep[indices,0] += ex[i]*bi
                Ep[indices,1] += ey[i]*bi
                Bp[indices,0] += bx[i]*bi
                Bp[indices,1] += by[i]*bi
       
            return Ep,Bp
