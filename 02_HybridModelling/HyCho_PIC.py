from pyccel.decorators import types
from pyccel.decorators import pure
from pyccel.decorators import external_call



#==============================================================================
@pure
@types('double[:]','double[:]','double[:]')
def cross(a, b, r):
    r[0] = a[1]*b[2] - a[2]*b[1]
    r[1] = a[2]*b[0] - a[0]*b[2]
    r[2] = a[0]*b[1] - a[1]*b[0]
#==============================================================================



#==============================================================================
@pure
@types('double[:]','int','double')
def find_span(knots, degree, x):
    
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots) - 1 - degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: 
        returnVal = low
    elif x >= knots[high]: 
        returnVal = high - 1
    else:
        
        # Perform binary search
        span = (low + high)//2
        
        while x < knots[span] or x >= knots[span + 1]:
            
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low + high)//2
        
        returnVal = span

    return returnVal
#============================================================================== 



#==============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs(knots, degree, x, span, left, right, values):
    
    left [:]  = 0.
    right[:]  = 0.

    values[0] = 1.
    
    for j in range(degree):
        left [j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved    = 0.
        for r in range(j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r]*temp
            saved     = left[j - r]*temp
        
        values[j + 1] = saved
#==============================================================================



#==============================================================================
@external_call
@types('double[:]','double[:,:](order=F)','double[:]','int','int[:]','double[:]','double[:]','int','int')
def current(particles_pos, particles_v_w, knots, degree, spans, jh_x, jh_y, n_base, rel):
    
    jh_x[:] = 0.
    jh_y[:] = 0.

    
    # ... needed for splines evaluation
    from numpy import empty
    from numpy import zeros
    from numpy import sqrt   

    left   = empty(degree,     dtype=float)
    right  = empty(degree,     dtype=float)
    values = zeros(degree + 1, dtype=float)
    # ...
    
    
    np = len(particles_pos)
    
    #$ omp parallel
    #$ omp do reduction (+ : jh_x, jh_y) private (ip, pos, span, left, right, values, ux, uy, uz, wp_over_gamma, il, i, bi )
    for ip in range(np):
        pos  = particles_pos[ip]
        span = spans[ip]
        basis_funs(knots, degree, pos, span, left, right, values)

        ux = particles_v_w[ip, 0]
        uy = particles_v_w[ip, 1]
        uz = particles_v_w[ip, 2]
        
        if rel == 1:
            wp_over_gamma = particles_v_w[ip, 3]/sqrt(1. + ux**2 + uy**2 + uz**2)
        else:
            wp_over_gamma = particles_v_w[ip, 3]

        for il in range(degree + 1):
            i  = (span - il)%n_base
            bi = values[degree - il]*wp_over_gamma

            jh_x[i] -= ux*bi
            jh_y[i] -= uy*bi

    #$ omp end do
    #$ omp end parallel
    
    jh_x = jh_x/np
    jh_y = jh_y/np
    
    ierr = 0
#==============================================================================
 
    
    
#==============================================================================
@external_call
@types('double[:,:](order=F)','double','double[:]','double[:]','int','int[:]','double','int','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','double[:,:](order=F)','int')
def pusher_periodic(particles, dt, t0, t1, p0, spans0, L, nbase0, ex, ey, bx, by, pp0, pp1, rel):
    
    from numpy import empty
    from numpy import zeros
    from numpy import sqrt
    
    p1         = p0 - 1
    
    E          = zeros(3     , dtype=float)
    B          = zeros(3     , dtype=float)
    
    delta      = L/nbase0
    
    qprime     = -dt/2
    
    u_minus    = zeros(3, dtype=float)
    t          = zeros(3, dtype=float)
    u_minus_xt = zeros(3, dtype=float)
    u_prime    = zeros(3, dtype=float)
    S          = zeros(3, dtype=float)
    u_prime_xS = zeros(3, dtype=float)
    u_plus     = zeros(3, dtype=float)
    
    np         = len(particles[:, 0])
    
    
    #$ omp parallel
    #$ omp do private (ip, pos_loc, span0, span1, E, B, u_minus, gamma, t, u_minus_xt, u_prime, normB, r, S, u_prime_xS, u_plus, il, jl, i, basis, power)
    for ip in range(np):
        
        # ... field evaluation (wave + background)
        span0   = spans0[ip]
        span1   = span0 - 1
        
        E[:]    = 0.
        B[:]    = 0.
        B[2]    = 1.
        
        pos_loc = particles[ip, 0] - (span0 - p0)*delta

        for jl in range(p0 + 1):
            power = pos_loc**jl

            for il in range(p0 + 1):
                i = (span0 - il)%nbase0
                basis = pp0[p0 - il, jl]*power

                E[0] += ex[i]*basis
                E[1] += ey[i]*basis


        for jl in range(p1 + 1):
            power = pos_loc**jl

            for il in range(p1 + 1):
                i = (span1 - il)%nbase0
                basis = pp1[p1 - il, jl]*power

                B[0] += bx[i]*basis
                B[1] += by[i]*basis
        # ...
        
        
        
        # ... Boris push (relativistic)
        if rel == 1:
            u_minus = particles[ip, 1:4] + qprime*E
            gamma   = sqrt(1. + u_minus[0]**2 + u_minus[1]**2 + u_minus[2]**2)
            t       = qprime/gamma*B
            cross(u_minus, t, u_minus_xt)
            u_prime = u_minus + u_minus_xt
            normB = B[0]**2 + B[1]**2 + B[2]**2
            r = 1 + (qprime/gamma)**2*normB
            S = 2*(qprime/gamma)*B/r
            cross(u_prime, S, u_prime_xS)
            u_plus = u_minus + u_prime_xS
            particles[ip, 1:4] = u_plus + qprime*E
            gamma = sqrt(1. + particles[ip, 1]**2 + particles[ip, 2]**2 + particles[ip, 3]**2)
            particles[ip, 0] += dt*particles[ip, 3]/gamma
            particles[ip, 0] = particles[ip, 0]%L
        # ...
        
        # ... Boris push (non-relativistic)
        else:
            u_minus = particles[ip, 1:4] + qprime*E
            t = qprime*B
            cross(u_minus, t, u_minus_xt)
            u_prime = u_minus + u_minus_xt
            normB = B[0]**2 + B[1]**2 + B[2]**2
            r = 1 + qprime**2*normB
            S = 2*qprime*B/r
            cross(u_prime, S, u_prime_xS)
            u_plus = u_minus + u_prime_xS
            particles[ip, 1:4] = u_plus + qprime*E
            particles[ip, 0] += dt*particles[ip, 3]
            particles[ip, 0] = particles[ip, 0]%L
        # ...
        
    #$ omp end do
    #$ omp end parallel
        
    ierr = 0
#==============================================================================
    
    
    
    
    
    
    
#==============================================================================
@external_call
@types('double[:,:](order=F)','double','double[:]','double[:]','int','int[:]','double','double','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','double[:,:](order=F)','double','int')
def pusher_reflecting(particles, dt, t0, t1, p0, spans0, L, delta, ex, ey, bx, by, pp0, pp1, xi, rel):
    
    from numpy import empty
    from numpy import zeros
    from numpy import sqrt
    
    p1         = p0 - 1

    Nl         = empty(p0,     dtype=float)
    Nr         = empty(p0,     dtype=float)
    N          = zeros(p0 + 1, dtype=float)
    
    Dl         = empty(p1    , dtype=float)
    Dr         = empty(p1    , dtype=float)
    D          = zeros(p1 + 1, dtype=float)
    
    E          = zeros(3     , dtype=float)
    B          = zeros(3     , dtype=float)
    
    vec_z      = zeros(3     , dtype=float)
    rho        = zeros(3     , dtype=float)
    vec_z[2]   = 1.
    
    
    b_left     = p0*delta
    b_right    = L - b_left
    
    
    qprime     = -dt/2
    
    u_minus    = zeros(3, dtype=float)
    t          = zeros(3, dtype=float)
    u_minus_xt = zeros(3, dtype=float)
    u_prime    = zeros(3, dtype=float)
    S          = zeros(3, dtype=float)
    u_prime_xS = zeros(3, dtype=float)
    u_plus     = zeros(3, dtype=float)
    
    np         = len(particles[:, 0])
    
    
    #$ omp parallel
    #$ omp do private (ip, pos, pos_loc, span0, span1, Nl, Nr, N, Dl, Dr, D, E, B, rho, u_minus, gamma, t, u_minus_xt, u_prime, normB, r, S, u_prime_xS, u_plus, il, jl, i, basis, power)
    for ip in range(np):
        
        # ... field evaluation (wave + background)
        pos    = particles[ip, 0]
        span0  = spans0[ip]
        span1  = span0 - 1
        
        E[:]   = 0.
        B[:]   = 0.
        B[2]   = 1. + xi*(pos - L/2)**2
        rho[:] = 0.
            
       
        if (pos < b_left) or (pos > b_right):
            basis_funs(t0, p0, pos, span0, Nl, Nr, N)
            basis_funs(t1, p1, pos, span1, Dl, Dr, D)

            for il in range(p0 + 1):
                i = span0 - il
                
                basis = N[p0 - il]
                
                E[0] += ex[i]*basis
                E[1] += ey[i]*basis
                
            for il in range(p1 + 1):
                i = span1 - il
                
                basis = D[p1 - il]*p0/(t1[i + p0] - t1[i])
                
                B[0] += bx[i]*basis
                B[1] += by[i]*basis       
        else:
            
            pos_loc = pos - (span0 - p0)*delta
            
            for jl in range(p0 + 1):
                power = pos_loc**jl
                
                for il in range(p0 + 1):
                    i = span0 - il
                    basis = pp0[p0 - il, jl]*power
                    
                    E[0] += ex[i]*basis
                    E[1] += ey[i]*basis
                    
            
            for jl in range(p1 + 1):
                power = pos_loc**jl
                
                for il in range(p1 + 1):
                    i = span1 - il
                    basis = pp1[p1 - il, jl]*power
                    
                    B[0] += bx[i]*basis
                    B[1] += by[i]*basis
        
        
        cross(particles[ip, 1:4], vec_z, rho)
        B[0] -= rho[0]/B[2]*(pos - L/2)*xi
        B[1] -= rho[1]/B[2]*(pos - L/2)*xi
        # ...
        
        
        
        # ... Boris push (relativistic)
        if rel == 1:
            u_minus = particles[ip, 1:4] + qprime*E
            gamma = sqrt(1. + u_minus[0]**2 + u_minus[1]**2 + u_minus[2]**2)
            t = qprime/gamma*B
            cross(u_minus, t, u_minus_xt)
            u_prime = u_minus + u_minus_xt
            normB = B[0]**2 + B[1]**2 + B[2]**2
            r = 1 + (qprime/gamma)**2*normB
            S = 2*(qprime/gamma)*B/r
            cross(u_prime, S, u_prime_xS)
            u_plus = u_minus + u_prime_xS
            particles[ip, 1:4] = u_plus + qprime*E
            gamma = sqrt(1. + particles[ip, 1]**2 + particles[ip, 2]**2 + particles[ip, 3]**2)
            particles[ip, 0] += dt*particles[ip, 3]/gamma
            
            if (particles[ip, 0] > L) or (particles[ip, 0] < 0.):
                particles[ip, 3] = -particles[ip, 3]
                particles[ip, 0] = pos + dt*particles[ip, 3]/gamma
                    
        # ...
        
        # ... Boris push (non-relativistic)
        else:
            u_minus = particles[ip, 1:4] + qprime*E
            t = qprime*B
            cross(u_minus, t, u_minus_xt)
            u_prime = u_minus + u_minus_xt
            normB = B[0]**2 + B[1]**2 + B[2]**2
            r = 1 + qprime**2*normB
            S = 2*qprime*B/r
            cross(u_prime, S, u_prime_xS)
            u_plus = u_minus + u_prime_xS
            particles[ip, 1:4] = u_plus + qprime*E
            particles[ip, 0] += dt*particles[ip, 3]
            
            if (particles[ip, 0] > L) or (particles[ip, 0] < 0.):
                particles[ip, 0] = pos - dt*particles[ip, 3]
        # ...
        
    #$ omp end do
    #$ omp end parallel
        
    ierr = 0
#==============================================================================