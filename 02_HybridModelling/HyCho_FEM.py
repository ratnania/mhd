import numpy as np
import bsplines as bsp
import scipy.sparse as spa
import scipy.special as sp

#================================================== mass matrix in V0 =========================================================
def mass_V0(T, p, bc):
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b) - 1
    NbaseN           = Nel + p - bc*p
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisN           = bsp.basis_ders_on_quad_grid(T, p, pts, 0)

    M                = np.zeros((NbaseN, 2*p + 1))

    for ie in range(Nel):

        for il in range(p + 1):
            for jl in range(p + 1):

                value = 0.

                for q in range(p + 1):
                    value += wts[ie, q] * basisN[ie, il, 0, q] * basisN[ie, jl, 0, q]

                M[(ie + il)%NbaseN, p + jl - il] += value
                
    indices = np.indices((NbaseN, 2*p + 1))
    shift   = np.arange(NbaseN) - p
    
    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseN
    
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN, NbaseN))
    M.eliminate_zeros()
                
    return M    
#==============================================================================================================================



#================================================== mass matrix in V1 =========================================================
def mass_V1(T, p, bc):
    
    t                = T[1:-1]
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b) - 1
    NbaseN           = Nel + p - bc*p
    NbaseD           = NbaseN - (1 - bc)
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisD           = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)
    
    M                = np.zeros((NbaseD, 2*p + 1))

    for ie in range(Nel):

        for il in range(p):
            for jl in range(p):

                value = 0.

                for q in range(p + 1):
                    value += wts[ie, q] * basisD[ie, il, 0, q] * basisD[ie, jl, 0, q]

                M[(ie + il)%NbaseD, p + jl - il] += value
                
    
    indices = np.indices((NbaseD, 2*p + 1))
    shift   = np.arange(NbaseD) - p

    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseD
    
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD, NbaseD))
    M.eliminate_zeros()
                
    return M
#==============================================================================================================================



#=========================== mass matrix in V0 (weighted with background field ================================================
def mass_V0_B(T, p, bc, Bz0):
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b) - 1
    NbaseN           = Nel + p - bc*p
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisN           = bsp.basis_ders_on_quad_grid(T, p, pts, 0)

    M                = np.zeros((NbaseN, 2*p + 1))

    for ie in range(Nel):

        for il in range(p + 1):
            for jl in range(p + 1):

                value = 0.

                for q in range(p + 1):
                    value += wts[ie, q] * basisN[ie, il, 0, q] * basisN[ie, jl, 0, q] * Bz0(pts[ie, q])

                M[(ie + il)%NbaseN, p + jl - il] += value
                
    indices = np.indices((NbaseN, 2*p + 1))
    shift   = np.arange(NbaseN) - p
    
    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseN
    
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN, NbaseN))
    M.eliminate_zeros()
                
    return M    
#==============================================================================================================================



#================================================== L2 product in V0 ==========================================================
def L2_prod_V0(T, p, bc, fun):
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b) - 1
    NbaseN           = Nel + p - bc*p
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisN           = bsp.basis_ders_on_quad_grid(T, p, pts, 0)
    
    f_int            = np.zeros(NbaseN)
    
    for ie in range(Nel):
        for il in range(p + 1):
            
            value = 0.
            for q in range(p + 1):
                value += wts[ie, q] * fun(pts[ie, q]) * basisN[ie, il, 0, q]
                
            f_int[(ie + il)%NbaseN] += value
            
    return f_int
#==============================================================================================================================



#================================================== L2 product in V1 ==========================================================
def L2_prod_V1(T, p, bc, fun):
    
    t                = T[1:-1]
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b) - 1
    NbaseN           = Nel + p - bc*p
    NbaseD           = NbaseN - (1 - bc)
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisD           = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)
    
    f_int            = np.zeros(NbaseD)
    
    for ie in range(Nel):
        for il in range(p):
            
            value = 0.
            for q in range(p + 1):
                value += wts[ie, q]*fun(pts[ie, q])*basisD[ie, il, 0, q]
                
            f_int[(ie + il)%NbaseD] += value
            
    return f_int
#==============================================================================================================================



#================================================ FEM field in V0 =============================================================
def FEM_field_V0(coeff, x, T, p, bc):
    
    N = bsp.collocation_matrix(T, p, x, bc)          
    eva = np.dot(N, coeff) 
        
    return eva
#==============================================================================================================================



#================================================ FEM field in V0 =============================================================
def FEM_field_V1(coeff, x, T, p, bc):
    
    t = T[1:-1]
    D = bsp.collocation_matrix(t, p - 1, x, bc, normalize=True)
    eva = np.dot(D, coeff)
    
    return eva
#==============================================================================================================================



#============================================ discrete gradient matrix (V0 --> V1) ============================================
def GRAD(T, p, bc):
    
    el_b      = bsp.breakpoints(T, p)
    Nel       = len(el_b) - 1
    NbaseN    = Nel + p - bc*p
    
    
    if bc == True:
        
        G = np.zeros((NbaseN, NbaseN))
        
        for i in range(NbaseN):
            
            G[i, i] = -1.
            
            if i < NbaseN - 1:
                G[i, i + 1] = 1.
                
        G[-1, 0] = 1.
        
        return G
    
    
    else:
        
        G = np.zeros((NbaseN - 1, NbaseN))
    
        for i in range(NbaseN - 1):
            
            G[i, i] = -1.
            G[i, i  + 1] = 1.
            
        return G
#==============================================================================================================================



#============================================================================================================================== 
def integrate_1d(points, weights, fun):
    
    n = points.shape[0]
    k = points.shape[1]

    f_int = np.zeros(n)
    
    for ie in range(n):
        for g in range(k):
            f_int[ie] += weights[ie, g]*fun(points[ie, g])
        
    return f_int
#==============================================================================================================================



#==============================================================================================================================
def histo(T, p, bc, grev):
    
    el_b = bsp.breakpoints(T, p)
    Nel = len(el_b) - 1
    t = T[1:-1]
    Nbase = Nel + p - bc*p
    
    if bc == True:
        
        pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
        
        ne = Nbase
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
#==============================================================================================================================



#==============================================================================================================================
class projectors_1d:
    
    def __init__(self, T, p, bc):
        
        self.p     = p
        self.T     = T
        self.bc    = bc
        
        self.greville    = bsp.greville(T, p, bc)
        self.breakpoints = bsp.breakpoints(T, p)
        self.Nbase       = len(self.breakpoints) - 1 + p - bc*p
        
        self.pts_loc, self.wts_loc = np.polynomial.legendre.leggauss(p)
        
        self.interpolation_V0 = bsp.collocation_matrix(T, p, self.greville, bc)  
        self.histopolation_V1 = histo(T, p, bc, self.greville)
        
        
        
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
#============================================================================================================================== 
    
    
    
#==============================================================================================================================
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
#==============================================================================================================================



#==============================================================================================================================
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
#==============================================================================================================================



#==============================================================================================================================
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
#==============================================================================================================================



#==============================================================================================================================
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
#==============================================================================================================================