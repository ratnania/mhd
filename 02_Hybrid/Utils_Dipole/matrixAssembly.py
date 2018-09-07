# ... assembles the mass, convection and background field matrix of a B-spline Finite Element basis of degree p

import numpy as np

def matrixAssembly(basis,p,Nz,weights,quad_points,B0z,xi,Lz):
	
    Nb = Nz
    M = np.zeros((Nb,Nb))
    C = np.zeros((Nb,Nb))
    D = np.zeros((Nb,Nb))


    for ie in range(0,Nz):
        for il in range(0,p+1):
            for jl in range(0,p+1):
                
                i = il + ie  
                j = jl + ie
                
                value_m = 0.0
                value_c = 0.0
                value_d = 0.0
                
                for g in range(0,p+1):
                    gl = ie*(p+1) + g
                    
                    value_m += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)
                    value_c += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,1)
                    value_d += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)*B0z*(1 + xi*(quad_points[gl] - Lz/2)**2)
                    
                M[i%Nb,j%Nb] += value_m
                C[i%Nb,j%Nb] += value_c
                D[i%Nb,j%Nb] += value_d

    return M,C,D
