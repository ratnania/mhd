import numpy as np
import scipy.sparse as sparse
import scipy.special as sp
import bsplines as bsp


#===============================================================================================================
def matrixAssembly_V0(p, Nbase, T, bc):
    """
    Computes the 1d mass and advection matrix in the space V0.
    
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
    
    
    el_b = bsp.breakpoints(T, p)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    d = 1
    basis = bsp.basis_ders_on_quad_grid(T, p, pts, d)

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
                    value_m += wts[ie, g]*basis[ie, il, 0, g]*basis[ie, jl, 0, g]
                    value_c += wts[ie, g]*basis[ie, il, 0, g]*basis[ie, jl, 1, g]

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
def matrixAssembly_V1(p, Nbase, T, bc):
    """
    Computes the 1d mass matrix in the space V1.
    
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
        mass matrix in V1
    """
    
    t = T[1:-1]
    
    el_b = bsp.breakpoints(T, p)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    d = 0
    basis = bsp.basis_ders_on_quad_grid(t, p - 1, pts, d, normalize=True)

    M = np.zeros((Nbase - 1, Nbase - 1))

    for ie in range(ne):
        for il in range(p):
            for jl in range(p):
                i = ie + il
                j = ie + jl

                value_m = 0.

                for g in range(p + 1):
                    value_m += wts[ie, g]*basis[ie, il, 0, g]*basis[ie, jl, 0, g]

                M[i, j] += value_m
                
    if bc == True:
        M[:p - 1, :] += M[-p + 1:, :]
        M[:, :p - 1] += M[:, -p + 1:]
        M = M[:M.shape[0] - p + 1, :M.shape[1] - p + 1]
                               
    return M




#===============================================================================================================
def matrixAssembly_backgroundField(p, Nbase, T, bc, B_background_z):
    """
    Computes the 1d mass matrix weighted with some background field in the space V0.
    
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
    
    el_b = bsp.breakpoints(T, p)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    d = 0
    basis = bsp.basis_ders_on_quad_grid(T, p, pts, d)

    D = np.zeros((Nbase, Nbase))

    for ie in range(ne):
        for il in range(p + 1):
            for jl in range(p + 1):
                i = ie + il
                j = ie + jl

                value_d = 0.

                for g in range(p + 1):
                    value_d += wts[ie, g]*basis[ie, il, 0, g]*basis[ie, jl, 0, g]*B_background_z(pts[ie, g])

                D[i, j] += value_d
                
    if bc == True:
        D[:p, :] += D[-p:, :]
        D[:, :p] += D[:, -p:]
        D = D[:D.shape[0] - p, :D.shape[1] - p]
        
    elif bc == False:
        D = D[1:-1, 1:-1]
        
                               
    return D



#===============================================================================================================
def L2_prod_V0(fun, p, Nbase, T):
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
    
    
    el_b = bsp.breakpoints(T, p)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    d = 0
    basis = bsp.basis_ders_on_quad_grid(T, p, pts, d)
    
    f_int = np.zeros(Nbase)
    
    for ie in range(ne):
        for il in range(p + 1):
            i = ie + il
            
            value = 0.
            for g in range(p + 1):
                value += wts[ie, g]*fun(pts[ie, g])*basis[ie, il, 0, g]
                
            f_int[i] += value
            
    return f_int



#===============================================================================================================
def L2_prod_V1(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V1
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
    
    t = T[1:-1]
    
    el_b = bsp.breakpoints(T, p)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    d = 0
    basis = bsp.basis_ders_on_quad_grid(t, p - 1, pts, d, normalize=True)
    
    f_int = np.zeros(Nbase - 1)
    
    for ie in range(ne):
        for il in range(p):
            i = ie + il
            
            value = 0.
            for g in range(p + 1):
                value += wts[ie, g]*fun(pts[ie, g])*basis[ie, il, 0, g]
                
            f_int[i] += value
            
    return f_int



#===============================================================================================================
def evaluate_field_V0(vec, x, p, Nbase, T, bc):
    """
    Evaluates the 1d FEM field in the space V0 at the points x.
    
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
        N = bsp.collocation_matrix(T, p, x, bc)
              
    elif bc == False:
        N = bsp.collocation_matrix(T, p, x, bc)[:, 1:-1]
        
    else:    
        N = bsp.collocation_matrix(T, p, x, False)
        
    eva = np.dot(N, vec) 
        
    return eva



#===============================================================================================================
def evaluate_field_V1(vec, x, p, Nbase, T, bc):
    """
    Evaluates the 1d FEM field in the space V1 at the points x.
    
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
    eval: np.arrray
        the function values at the points x
    """
    
    t = T[1:-1]
    
    if bc == True:
        D = bsp.collocation_matrix(t, p - 1, x, bc, normalize=True)
        
    else:
        D = bsp.collocation_matrix(t, p - 1, x, False, normalize=True)
    
    eva = D.dot(vec)
    
    return eva



#===============================================================================================================
def GRAD_1d(p, Nbase, bc):
    """
    Returns the 1d discrete gradient matrix.
    
    Parameters
    ----------
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    Returns
    -------
    G: 2d np.array
        discrete gradient matrix
    """
    
    if bc == True:
        Nbase_0 = Nbase - p
        
        G = np.zeros((Nbase_0, Nbase_0))
        
        for i in range(Nbase_0):
            G[i, i] = -1.
            if i < Nbase_0 - 1:
                G[i, i + 1] = 1.
                
        G[-1, 0] = 1.
        
        return G
    
    else:
        
        G = np.zeros((Nbase - 1, Nbase))
    
        for i in range(Nbase - 1):
            G[i, i] = -1.
            G[i, i  + 1] = 1.
            
        if bc == False:
            G = G[:, 1:-1]

        return G
    
    
    
#=============================================================================================================== 
def integrate_1d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 1d.
    
    Parameters
    ----------
    points : 2d np.array
        quadrature points in format (local point, element)
        
    weights : 2d np.array
        quadrature weights in format (local point, element)
    
    fun : callable
        1d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    n = points.shape[0]
    k = points.shape[1]
    
    
    f_int = np.zeros(n)
    
    for ie in range(n):
        for g in range(k):
            f_int[ie] += weights[ie, g]*fun(points[ie, g])
        
    return f_int




#===============================================================================================================
def histopolation_matrix_1d(p, Nbase, T, grev, bc):
    """
    Computest the 1d histopolation matrix.
    
    Parameters
    ----------
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
    
    T : np.array 
        knot vector
    
    grev : np.array
        greville points
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    Returns
    -------
    D : 2d np.array
        histopolation matrix
    """
    
    el_b = bsp.breakpoints(T, p)
    Nel = len(el_b) - 1
    t = T[1:-1]
    
    if bc == True:
        
        pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
        
        ne = Nbase - p
        D = np.zeros((ne, ne))
        
        if p%2 != 0:
            
            grid = el_b
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            col_quad = bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=True)
            
            for ie in range(Nel):
                for il in range(p):
                    
                    i = (ie + il)%ne
                    
                    for k in range(p - 1):
                        D[ie, i] += wts[ie, k]*col_quad[ie*(p - 1) + k, i]

            return D
        
        else:
            
            grid = np.linspace(0., el_b[-1], 2*Nel + 1)
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            col_quad = bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=True)
            
            for iee in range(2*Nel):
                for il in range(p):
                    
                    ie = int(iee/2)
                    ie_grev = int(np.ceil(iee/2) - 1)
                    
                    i = (ie + il)%ne
                    
                    for k in range(p - 1):
                        D[ie_grev, i] += wts[iee, k]*col_quad[iee*(p - 1) + k, i]

            return D
        
    else:
        
        ng = len(grev)
        
        col_quad = bsp.collocation_matrix(T, p, grev, bc)
        
        D = np.zeros((ng - 1, Nbase - 1))
        
        for i in range(ng - 1):
            for j in range(max(i - p + 1, 1), min(i + p + 3, Nbase)):
                s = 0.
                for k in range(j):
                    s += col_quad[i, k] - col_quad[i + 1, k]
                    
                D[i, j - 1] = s
                
        return D
        


class projectors_1d:
    
    def __init__(self, p, Nbase, T, bc):
        
        self.p  = p
        self.T  = T
        self.bc = bc
        
        self.greville    = bsp.greville(T, p, bc)
        self.breakpoints = bsp.breakpoints(T, p)
        
        self.pts_loc, self.wts_loc = np.polynomial.legendre.leggauss(p)
        
        self.interpolation_V0 = bsp.collocation_matrix(T, p, self.greville, bc)  
        self.histopolation_V1 = histopolation_matrix_1d(p, Nbase, T, self.greville, bc)
        
        
        
    def PI_0(self, fun):
        
        # Assemble vector of interpolation problem at greville points
        rhs = fun(self.greville)
        
        # Solve interpolation problem
        vec = np.linalg.solve(self.interpolation_V0, rhs)
        
        return vec
    
    
    
    def PI_1(self, fun):
        
        # Compute quadrature grid for integration of right-hand-side
        if self.bc == True:
            if self.p%2 != 0:

                grid = self.breakpoints
                pts, wts = bsp.quadrature_grid(grid, self.pts_loc, self.wts_loc)

            else:

                grid = np.append(self.greville, self.greville[-1] + (self.greville[-1] - self.greville[-2])) 
                pts, wts = bsp.quadrature_grid(grid, self.pts_loc, self.wts_loc)%self.breakpoints[-1]

        else:

            pts, wts = bsp.quadrature_grid(self.greville, self.pts_loc, self.wts_loc)
            
        # Assemble vector of histopolation problem at greville points
        rhs = integrate_1d(pts, wts, fun)
        
        # Solve histopolation problem
        vec = np.linalg.solve(self.histopolation_V1, rhs)
        
        return vec   
    
    
    
#===============================================================================================================
def solveDispersionHybrid(k, pol, c, wce, wpe, wpar, wperp, nuh, initial_guess, tol, max_it=100):
    """
    Solves the hybrid dispersion relation numerically using Newton's method for a fixed wavenumber k.
    
    Parameters
    ----------
    k : double
        wavenumber
        
    pol : int
        wave polarization (+1 : R-wave, -1: L-wave)
        
    c : double
        the speed of light
        
    wce : double
        electron cyclotron frequency
        
    wpe : double
        cold electron plasma frequency
        
    wpar : double
        parallel thermal velocity of energetic electrons
        
    wperp : double
        perpendicular thermal velocity of energetic electrons
        
    nuh : double
        ratio between cold and energetic electron number densities
        
    initial_guess : complex
        initial guess of solution for initialization of Newton's method
        
    tol : double
        tolerance when to stop Newton interation (difference between two successive interations)
        
    max_it : int
        maximum number of iterations
        
    Returns
    -------
    
    w : complex
        solution (frequency) of dispersion realation
        
    counter : int
        number of needed iterations
    """

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
def solveDispersionHybridExplicit(k, pol, c, wce, wpe, wpar, wperp, nuh, initial_guess, tol, max_it=100):
    """
    Solves the hybrid dispersion relation numerically using Newton's method for a fixed wavenumber k.
    
    Parameters
    ----------
    k : double
        wavenumber
        
    pol : int
        wave polarization (+1 : R-wave, -1: L-wave)
        
    c : double
        the speed of light
        
    wce : double
        electron cyclotron frequency
        
    wpe : double
        cold electron plasma frequency
        
    wpar : double
        parallel thermal velocity of energetic electrons
        
    wperp : double
        perpendicular thermal velocity of energetic electrons
        
    nuh : double
        ratio between cold and energetic electron number densities
        
    initial_guess : complex
        initial guess of solution for initialization of Newton's method
        
    tol : double
        tolerance when to stop Newton interation (difference between two successive interations)
        
    max_it : int
        maximum number of iterations
        
    Returns
    -------
    
    wr : double
        real frequency of dispersion realation
        
    wi : double
        growth rate
        
    counter : int
        number of needed iterations
    """

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
def solveDispersionHybridRelativistic(k, pol, c, wce, wpe, wpar, wperp, nuh, initial_guess, tol, max_it=100):
    """
    Solves the relativistic hybrid dispersion relation numerically using Newton's method for a fixed wavenumber k.
    
    Parameters
    ----------
    k : double
        wavenumber
        
    pol : int
        wave polarization (+1 : R-wave, -1: L-wave)
        
    c : double
        the speed of light
        
    wce : double
        electron cyclotron frequency
        
    wpe : double
        cold electron plasma frequency
        
    wpar : double
        parallel thermal velocity of energetic electrons
        
    wperp : double
        perpendicular thermal velocity of energetic electrons
        
    nuh : double
        ratio between cold and energetic electron number densities
        
    initial_guess : complex
        initial guess of solution for initialization of Newton's method
        
    tol : double
        tolerance when to stop Newton interation (difference between two successive interations)
        
    max_it : int
        maximum number of iterations
        
    Returns
    -------
    
    wr : double
        real frequency of dispersion realation
        
    wi : double
        growth rate
        
    counter : int
        number of needed iterations
    """
    
    import scipy.integrate as integrate

    def Dcold(k, w, pol):
        return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce))

    def Dcoldprime(k, w, pol):
        return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2)
    
    def gamma(p_perp, k, om):
        return (-1 + (c*k/om)*(((c*k/om)**2 - 1)*(1 + p_perp**2/c**2)*(om/np.abs(wce))**2 + 1)**(1/2))/(((c*k/om)**2 - 1)*(om/np.abs(wce)))

    def pR(p_perp, k, om):
        return (gamma(p_perp, k, om)*om - np.abs(wce))/k

    def DeltaR(p_perp, k, om):
        return 1 - om*pR(p_perp, k, om)/(c**2*gamma(p_perp, k, om)*k)

    def integrand_eta(p_perp, k, om):
        return np.pi*nuh*(om - np.abs(wce))/k*(p_perp**2/DeltaR(p_perp, k, om))*(-p_perp/wperp**2)*1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-pR(p_perp, k, om)**2/(2*wpar**2) - p_perp**2/(2*wperp**2))

    def integrand_A1(p_perp, k, om):
        return k/(om - np.abs(wce))*p_perp**2/(DeltaR(p_perp, k, om)*gamma(p_perp, k, om))*1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-pR(p_perp, k, om)**2/(2*wpar**2) - p_perp**2/(2*wperp**2))*(pR(p_perp, k, om)*(p_perp/wperp**2) - p_perp*pR(p_perp, k, om)/wpar**2)

    def integrand_A2(p_perp, k, om):
        return p_perp**2/DeltaR(p_perp, k, om)*1/((2*np.pi)**(3/2)*wpar*wperp**2)*np.exp(-pR(p_perp, k, om)**2/(2*wpar**2) - p_perp**2/(2*wperp**2))*(-p_perp/wperp**2)

    wr = initial_guess
    counter = 0

    while True:
        wnew = wr - Dcold(k, wr, pol)/Dcoldprime(k, wr, pol)

        if np.abs(wnew - wr) < tol or counter == max_it:
            wr = wnew
            break

        wr = wnew
        counter += 1

    i1 = integrate.quad(integrand_eta, 0., np.inf, args=(k, wr))[0]
    i2 = integrate.quad(integrand_A1, 0., np.inf, args=(k, wr))[0]
    i3 = integrate.quad(integrand_A2, 0., np.inf, args=(k, wr))[0]
    
    wi = np.pi*wpe**2*i1/(2*wr + wpe**2*np.abs(wce)/(wr - np.abs(wce))**2)*(i2/i3 - wr/(np.abs(wce) - wr))

    return wr, wi, counter



#===============================================================================================================
def solveDispersionCold(k, pol, c, wce, wpe, initial_guess, tol, max_it = 100):
    """
    Solves the cold plasma dispersion relation numerically using Newton's method for a fixed wavenumber k.
    
    Parameters
    ----------
    k : double
        wavenumber
        
    pol : int
        wave polarization (+1 : R-wave, -1: L-wave)
        
    c : double
        the speed of light
        
    wce : double
        electron cyclotron frequency
        
    wpe : double
        cold electron plasma frequency
        
    initial_guess : complex
        initial guess of solution for initialization of Newton's method
        
    tol : double
        tolerance when to stop Newton interation (difference between two successive interations)
        
    max_it : int
        maximum number of iterations
        
    Returns
    -------
    
    w : complex
        solution (frequency) of dispersion realation
        
    counter : int
        number of needed iterations
    """

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