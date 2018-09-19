def evaluation(uj, basis , x, p, bcs = 1):
    ''' Given a coefficient vector, this function computes the value at some position in space spanned by a B-spline basis
    
        Parameters:
            uj : ndarray
                Coefficient vector.
            basis : Bspline object
                Object of B-spline basis functions. 
            x : ndarray
                Positions to be evaluated.
            bcs : int
                Boundary conditions. DEFAULT = 1 (periodic), bcs = 2 (Dirichlet).
                
                
        Returns:
            u : ndarray
                Vector with values at positions x
    '''
    
    
    p = basis.p
    
    if bcs == 1:
        
        N_base = len(uj) + p
        N_el = len(uj)
 
        u = 0

        for i in range(0,N_base):
            u += uj[i%N_el]*basis(x,i)
        
        return u

    elif bcs == 2:
        
        N_base = len(uj)
        
        u = 0
        
        for i in range(0,N_base):
            u += uj[i]*basis(x,i)
        
        return u
