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
@external_call
@types('double[:,:](order=F)','double[:]','int','int[:]','double[:]','double[:]','int')
def current(particles, knots, degree, spans, jh_x, jh_y, n_base):
    
    jh_x[:] = 0.
    jh_y[:] = 0.

    
    # ... needed for splines evaluation
    from numpy import empty
    from numpy import zeros
    from numpy import sqrt   

    left   = empty(degree,     dtype=float)
    right  = empty(degree,     dtype=float)
    values = zeros(degree + 1, dtype=float)

    span = 0
    # ...
    
    
    np = len(particles[:, 0])
    
    #$ omp parallel
    #$ omp do reduction ( + : jh_x, jh_y ) private ( ip, pos, span, left, right, values, ux, uy, uz, wp_over_gamma, il, i, bi )
    for ip in range(np):
        pos  = particles[ip, 0]
        span = spans[ip]
        basis_funs(knots, degree, pos, span, left, right, values)

        ux = particles[ip, 1]
        uy = particles[ip, 2]
        uz = particles[ip, 3]

        wp_over_gamma = particles[ip, 4]/sqrt(1. + ux**2 + uy**2 + uz**2)

        for il in range(degree + 1):
            i  = (span - il)%n_base
            bi = values[degree - il]*wp_over_gamma

            jh_x[i] -= ux*bi
            jh_y[i] -= uy*bi

    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    


#==============================================================================
@external_call
@types('double[:,:](order=F)','double','double[:]','double[:]','int','int[:]','double','double','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','double[:,:](order=F)','double')
def pusher_reflecting(particles, dt, t0, t1, p0, spans_0, L, delta, ex, ey, bx, by, pp_0, pp_1, xi):
    
    # ... needed for splines evaluation (in spaces V0 and V1)
    from numpy import empty
    from numpy import zeros
    from numpy import sqrt
    
    p1 = p0 - 1

    l0 = empty(p0,     dtype=float)
    r0 = empty(p0,     dtype=float)
    v0 = zeros(p0 + 1, dtype=float)
    
    l1 = empty(p1    , dtype=float)
    r1 = empty(p1    , dtype=float)
    v1 = zeros(p1 + 1, dtype=float)

    span_0 = 0
    span_1 = 0
    
    E   = zeros(3, dtype=float)
    B   = zeros(3, dtype=float)
    
    e_z = zeros(3, dtype=float)
    rho = zeros(3, dtype=float)
    
    e_z[2] = 1.
    # ...
    
    
    # ... define boundary region
    b_left = p0*delta
    b_right= L - b_left
    # ...
    
    
    # ... needed for Boris algorithm
    qprime = -dt/2
    
    u_minus    = zeros(3, dtype=float)
    t          = zeros(3, dtype=float)
    u_minus_xt = zeros(3, dtype=float)
    u_prime    = zeros(3, dtype=float)
    S          = zeros(3, dtype=float)
    u_prime_xS = zeros(3, dtype=float)
    u_plus     = zeros(3, dtype=float)
    # ...
    
    np = len(particles[:, 0])
    
    
    #$ omp parallel
    #$ omp do private ( ip, pos, span_0, l0, r0, v0, span_1, l1, r1, v1, E, B, rho, u_minus, gamma, t, u_minus_xt, u_prime, normB, r, S, u_prime_xS, u_plus, il0, jl0, i0, basis0, il1, jl1, i1, basis1 )
    for ip in range(np):
        # ... field evaluation (wave + background)
        pos    = particles[ip, 0]
        span_0 = spans_0[ip]
        span_1 = span_0 - 1
        
        E[:]   = 0.
        B[:]   = 0.
        B[2]   = 1. + xi*(pos - L/2)**2
        rho[:] = 0.
       
        if (pos < b_left) or (pos > b_right):
            basis_funs(t0, p0, pos, span_0, l0, r0, v0)
            basis_funs(t1, p1, pos, span_1, l1, r1, v1)

            for il0 in range(p0 + 1):
                i0 = span_0 - il0
                
                basis0 = v0[p0 - il0]
                
                E[0] += ex[i0]*basis0
                E[1] += ey[i0]*basis0
                
            for il1 in range(p1 + 1):
                i1 = span_1 - il1
                
                basis1 = v1[p1 - il1]*p0/(t1[i1 + p0] - t1[i1])
                
                B[0] += bx[i1]*basis1
                B[1] += by[i1]*basis1
                
            gamma = sqrt(1. + particles[ip, 1]**2 + particles[ip, 2]**2 + particles[ip, 3]**2)
            
            if (particles[ip, 0] + dt*particles[ip, 3]/gamma > L) or (particles[ip, 0] + dt*particles[ip, 3]/gamma < 0.):
                particles[ip, 3] = -particles[ip, 3]
                
        else:
            for il0 in range(p0 + 1):
                i0 = span_0 - il0
                
                for jl0 in range(p0 + 1):
                    
                    basis0 = pp_0[p0 - il0, jl0]*((pos - (span_0 - p0)*delta))**jl0
                    
                    E[0] += ex[i0]*basis0
                    E[1] += ey[i0]*basis0
                    
            for il1 in range(p1 + 1):
                i1 = span_1 - il1
                
                for jl1 in range(p1 + 1):
                    
                    basis1 = pp_1[p1 - il1, jl1]*((pos - (span_1 - p1)*delta))**jl1/delta
                    
                    B[0] += bx[i1]*basis1
                    B[1] += by[i1]*basis1
        
        cross(particles[ip, 1:4], e_z, rho)
        B[0] -= rho[0]/B[2]*(pos - L/2)*xi
        B[1] -= rho[1]/B[2]*(pos - L/2)*xi
        # ...
        
        
        # ... Boris push
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
        # ...
        
    #$ omp end do
    #$ omp end parallel
        
    ierr = 0