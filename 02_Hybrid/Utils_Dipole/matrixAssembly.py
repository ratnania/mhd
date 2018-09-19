# ... assembles the mass, convection and background field matrix of a B-spline Finite Element basis of degree p (periodic b.c: bcs = 1, Dirichlet b.c.: bcs = 2)

import numpy as np

def matrixAssembly(basis,p,N_el,weights,quad_points,B0,bcs = 1,damp = None,dz_damp = None):

    if bcs == 1:

        Nb = N_el
        M = np.zeros((Nb,Nb))
        C = np.zeros((Nb,Nb))
        D = np.zeros((Nb,Nb))


        for ie in range(0,N_el):
            for il in range(0,p + 1):
                for jl in range(0,p + 1):
                
                    i = il + ie  
                    j = jl + ie
                
                    value_m = 0.0
                    value_c = 0.0
                    value_d = 0.0
                
                    for g in range(0,p + 1):
                        gl = ie*(p + 1) + g
                    
                        value_m += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)
                        value_c += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,1)
                        value_d += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)*B0(quad_points[gl])
                    
                    M[i%Nb,j%Nb] += value_m
                    C[i%Nb,j%Nb] += value_c
                    D[i%Nb,j%Nb] += value_d

        return M,C,D

    elif bcs == 2:

        N = N_el + p

        M = np.zeros((N,N))
        C = np.zeros((N,N))
        D = np.zeros((N,N))

        #M_damp = np.zeros((N,N))
        #C_damp = np.zeros((N,N))
        #D_damp = np.zeros((N,N))
        #M_dz_damp = np.zeros((N,N))

        for ie in range(0,N_el):
            for il in range(0,p + 1):
                for jl in range(0,p + 1):
                
                    i = il + ie  
                    j = jl + ie
                
                    value_m = 0.0
                    value_c = 0.0
                    value_d = 0.0

                    #value_md = 0.0
                    #value_cd = 0.0
                    #value_dd = 0.0
                    #value_dz_md = 0.0
                
                    for g in range(0,p + 1):
                        gl = ie*(p + 1) + g
                    
                        value_m += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)
                        value_c += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,1)
                        value_d += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)*B0(quad_points[gl])
                        
                        #value_md += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)*damp(quad_points[gl])
                        #value_cd += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,1)*damp(quad_points[gl])
                        #value_dd += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)*B0(quad_points[gl])*damp(quad_points[gl])
                        #value_dz_md += weights[gl]*basis(quad_points[gl],i,0)*basis(quad_points[gl],j,0)*B0(quad_points[gl])*dz_damp(quad_points[gl])
                    
                    M[i,j] += value_m
                    C[i,j] += value_c
                    D[i,j] += value_d

                    #M_damp[i,j] += value_md
                    #C_damp[i,j] += value_cd
                    #D_damp[i,j] += value_dd
                    #M_dz_damp[i,j] += value_dz_md


        #return M,C,D,M_damp,C_damp,D_damp,M_dz_damp
        return M,C,D
