import numpy as np
import psydac.core.interface as inter
import scipy.sparse as sparse
import scipy.special as sp


#===============================================================================================================
def matrixAssembly(p, Nbase, T, bc):
    """
    Computes the 1d mass and advection matrix of the space V0.
    
    Parameters
    ----------
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    Returns
    -------
    M : 2d np.array
        mass matrix in V0
        
    D : 2d np.array
        advection matrix in V0
    """
    
    if bc == True: 
        bcon = 1
        Nbase_0 = Nbase - p
    else:
        bcon = 0
        Nbase_0 = Nbase
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = inter.construct_quadrature_grid(ne, p + 1, pts_loc, wts_loc, el_b)

    d = 1
    basis = inter.eval_on_grid_splines_ders(p, Nbase, p + 1, d, T, pts)

    M = np.zeros((Nbase, Nbase))
    C = np.zeros((Nbase, Nbase))

    for ie in range(ne):
        for il in range(p + 1):
            for jl in range(p + 1):
                i = ie + il
                j = ie + jl

                value_m = 0.
                value_c = 0.

                for g in range(p + 1):
                    value_m += wts[g, ie]*basis[il, 0, g, ie]*basis[jl, 0, g, ie]
                    value_c += wts[g, ie]*basis[il, 0, g, ie]*basis[jl, 1, g, ie]

                M[i, j] += value_m
                C[i, j] += value_c
                
    if bc == True:
        M[:p, :] += M[-p:, :]
        M[:, :p] += M[:, -p:]
        M = M[:M.shape[0] - p, :M.shape[1] - p]
        
        C[:p, :] += C[-p:, :]
        C[:, :p] += C[:, -p:]
        C = C[:C.shape[0] - p, :C.shape[1] - p]
        
    elif bc == False:
        M = M[1:-1, 1:-1]
        C = C[1:-1, 1:-1]
        
                               
    return M, C


#===============================================================================================================
def matrixAssembly_backgroundField(p, Nbase, T, bc, B_background_z):
    """
    Computes the 1d mass and advection matrix of the space V0.
    
    Parameters
    ----------
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    B_background_z : callable
        the variation of the background magnetic field in z-direction
        
    Returns
    -------
    D : 2d np.array
        mass matrix in V0
    """
    
    if bc == True: 
        bcon = 1
        Nbase_0 = Nbase - p
    else:
        bcon = 0
        Nbase_0 = Nbase
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = inter.construct_quadrature_grid(ne, p + 1, pts_loc, wts_loc, el_b)

    d = 0
    basis = inter.eval_on_grid_splines_ders(p, Nbase, p + 1, d, T, pts)

    D = np.zeros((Nbase, Nbase))

    for ie in range(ne):
        for il in range(p + 1):
            for jl in range(p + 1):
                i = ie + il
                j = ie + jl

                value_d = 0.

                for g in range(p + 1):
                    value_d += wts[g, ie]*basis[il, 0, g, ie]*basis[jl, 0, g, ie]*B_background_z(pts[g, ie])

                D[i, j] += value_d
                
    if bc == True:
        D[:p, :] += D[-p:, :]
        D[:, :p] += D[:, -p:]
        D = D[:D.shape[0] - p, :D.shape[1] - p]
        
    elif bc == False:
        D = D[1:-1, 1:-1]
        
                               
    return D



#===============================================================================================================
def L2_prod(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V0
    using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1 
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = inter.construct_quadrature_grid(ne, p + 1, pts_loc, wts_loc, el_b)
    
    d = 0
    basis = inter.eval_on_grid_splines_ders(p, Nbase, p + 1, d, T, pts)
    
    f_int = np.zeros(Nbase)
    
    for ie in range(ne):
        for il in range(p + 1):
            i = ie + il
            
            value = 0.
            for g in range(p + 1):
                value += wts[g, ie]*fun(pts[g, ie])*basis[il, 0, g, ie]
                
            f_int[i] += value
            
    return f_int



#===============================================================================================================
def evaluate_field(vec, x, p, Nbase, T, bc):
    """
    Evaluates the 1d FEM field of the space V0 at the points x.
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : np.array
        evaluation points
        
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    Returns
    -------
    eva: np.arrray
        the function values at the points x
    """
    
    if bc == True:
        N = inter.collocation_matrix(p, Nbase, T, x)
        N[:, :p] += N[:, -p:]
        N = sparse.csr_matrix(N[:, :N.shape[1] - p])
        
    elif bc == False:
        N = sparse.csr_matrix(inter.collocation_matrix(p, Nbase, T, x)[:, 1:-1])
        
    else:    
        N = sparse.csr_matrix(inter.collocation_matrix(p, Nbase, T, x))
        
    eva = N.dot(vec) 
        
    return eva



#===============================================================================================================
def solveDispersionHybrid(k, pol, c, wce, wpe, wpar, wperp, nuh, initial_guess, tol, max_it = 100):

    Taniso = 1 - wperp**2/wpar**2


    def Z(xi):
        return np.sqrt(np.pi)*np.exp(-xi**2)*(1j - sp.erfi(xi))

    def Zprime(xi):
        return -2*(1 + xi*Z(xi))


    def Dhybrid(k, w, pol):
        xi = (w + pol*wce)/(k*np.sqrt(2)*wpar)

        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce)) + nuh*wpe**2/w**2*(w/(k*np.sqrt(2)*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi)))

   
    def Dhybridprime(k, w, pol):
        xi = (w + pol*wce)/(k*np.sqrt(2)*wpar)
        xip = 1/(k*np.sqrt(2)*wpar)

        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2) - 2*nuh*wpe**2/w**3*(w/(np.sqrt(2)*k*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi))) + nuh*wpe**2/w**2*(1/(np.sqrt(2)*k*wpar)*Z(xi) + w/(np.sqrt(2)*k*wpar)*Zprime(xi)*xip - Taniso*(xip*Z(xi) + xi*Zprime(xi)*xip)) 

    w = initial_guess
    counter = 0

    while True:
        wnew = w - Dhybrid(k, w, pol)/Dhybridprime(k, w, pol)

        if np.abs(wnew - w) < tol or counter == max_it:
            w = wnew
            break

        w = wnew
        counter += 1

    return w, counter



#===============================================================================================================
def solveDispersionHybridExplicit(k, pol, c, wce, wpe, wpar, wperp, nuh, initial_guess, tol, max_it = 100):

    def Dcold(k, w, pol):
        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce))

    def Dcoldprime(k, w, pol):
        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2)

    wr = initial_guess
    counter = 0

    while True:
        wnew = wr - Dcold(k, wr, pol)/Dcoldprime(k, wr, pol)

        if np.abs(wnew - wr) < tol or counter == max_it:
            wr = wnew
            break

        wr = wnew
        counter += 1

    vR = (wr + pol*wce)/k

    wi = 1/(2*wr - pol*wpe**2*wce/(wr + pol*wce)**2)*np.sqrt(2*np.pi)*wpe**2*nuh*vR/wpar*np.exp(-vR**2/(2*wpar**2))*(wr/(2*(-pol*wce - wr)) + 1/2*(1 - wperp**2/wpar**2))

    return wr, wi, counter



#===============================================================================================================
def solveDispersionCold(k, pol, c, wce, wpe, initial_guess, tol, max_it = 100):

    def Dcold(k, w, pol):
        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce))

    def Dcoldprime(k, w, pol):
        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2)

    wr = initial_guess
    counter = 0

    while True:
        wnew = wr - Dcold(k, wr, pol)/Dcoldprime(k, wr, pol)

        if np.abs(wnew - wr) < tol or counter == max_it:
            wr = wnew
            break

        wr = wnew
        counter += 1

    return wr, counter



#===============================================================================================================
def solveDispersionArtificial(k, pol, c, wce, wpe, c1, c2, mu0, initial_guess, tol, max_it = 100):

    def Dart(k, w, pol):
        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce)) + 1/w*(1j*c1 - pol*c2)
    
    def Dartprime(k, w, pol):
        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2) - 1/w**2*(1j*c1 - pol*c2)

    w = initial_guess
    counter = 0

    while True:
        wnew = w - Dart(k, w, pol)/Dartprime(k, w, pol)

        if np.abs(wnew - w) < tol or counter == max_it:
            w = wnew
            break

        w = wnew
        counter += 1

    return w, counter
