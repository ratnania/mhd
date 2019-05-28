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
@pure
@types('double[:]','int','double')
def find_span(knots, degree, x):
    """
    Determine the knot span index at location x, given the B-Splines' knot sequence and polynomial degree.

    For a degree p, the knot span index i identifies the indices [i-p:i] of all p+1 non-zero basis functions at a
    given location x.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Location of interest.

    Returns
    -------
    span : int
        Knot span index.

    """
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots) - 1 - degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high - 1
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
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs(knots, degree, x, span, left, right, values):
    """
    Compute the non-vanishing B-splines at location x, given the knot sequence, polynomial degree and knot span.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    Results
    -------
    values : numpy.ndarray
        Values of p+1 non-vanishing B-Splines at location x.

    """
    # to avoid degree being intent(inout)
    # TODO improve
    p = degree

#    from numpy      import empty
#    left   = empty( p  , dtype=float )
#    right  = empty( p  , dtype=float )
    left[:] = 0.
    right[:] = 0.

    values[0] = 1.0
    for j in range(0, p):
        left [j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved    = 0.0
        for r in range(0, j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r]*temp
            saved     = left[j - r]*temp
        values[j + 1] = saved


#==============================================================================
@external_call
@types('double[:]','double[:,:](order=F)','double[:]','int','double','double[:]','double','int')
def hotCurrentRel_bc_1(particles_pos, particles_vel_w, knots, p, qe, jh, c, Nb):
    
    jh[:] = 0.

    # ... needed for splines evaluation
    from numpy      import empty
    from numpy      import zeros

    left   = empty(p,     dtype=float )
    right  = empty(p,     dtype=float )
    values = zeros(p + 1, dtype=float )

    span = 0
    # ...

    npart = len(particles_pos)

    #$ omp parallel
    #$ omp do reduction ( + : jh ) private ( ipart, pos, span, left, right, values, wk, vx, vy, vz, il, i, ii, bi )
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span(knots, p, pos)
        basis_funs(knots, p, pos, span, left, right, values)

        wk = particles_vel_w[ipart, 3]
        
        vx = particles_vel_w[ipart, 0]
        vy = particles_vel_w[ipart, 1]
        vz = particles_vel_w[ipart, 2]
        
        gamma = (1 + (vx**2 + vy**2 + vz**2)/c**2)**(1/2)

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p - il]

            jh[2*ii] += vx*wk*bi/gamma
            jh[2*ii + 1] += vy*wk*bi/gamma

    #$ omp end do
    #$ omp end parallel

    jh = qe/npart*jh
    

#==============================================================================
@external_call
@types('double[:]','double[:,:](order=F)','double[:]','int','double','double[:]','double')
def hotCurrentRel_bc_2(particles_pos, particles_vel_w, knots, p, qe, jh, c):
    
    jh[:] = 0.

    # ... needed for splines evaluation
    from numpy      import empty
    from numpy      import zeros

    left   = empty(p,     dtype=float )
    right  = empty(p,     dtype=float )
    values = zeros(p + 1, dtype=float )

    span = 0
    # ...

    npart = len(particles_pos)

    #$ omp parallel
    #$ omp do reduction ( + : jh ) private ( ipart, pos, span, left, right, values, wk, vx, vy, vz, il, i, bi )
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span(knots, p, pos)
        basis_funs(knots, p, pos, span, left, right, values)

        wk = particles_vel_w[ipart, 3]
        
        vx = particles_vel_w[ipart, 0]
        vy = particles_vel_w[ipart, 1]
        vz = particles_vel_w[ipart, 2]
        
        gamma = (1 + (vx**2 + vy**2 + vz**2)/c**2)**(1/2)

        for il in range(0, p + 1):

            i = span - il
            bi = values[p - il]

            jh[2*i] += vx*wk*bi/gamma
            jh[2*i + 1] += vy*wk*bi/gamma

    #$ omp end do
    #$ omp end parallel

    jh = qe/npart*jh

    
#==============================================================================
@external_call
@types('double[:,:](order=F)','double','double','double','double','double[:]','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','double','double')
def borisGemRel_bc_1(particles, dt, qe, me, Lz, T, tz, p, Nb, ex, ey, bx, by, B0z, c):

    from numpy      import empty
    from numpy      import zeros

    l0 = empty(p,     dtype=float)
    r0 = empty(p,     dtype=float)
    v0 = zeros(p + 1, dtype=float)
    
    l1 = empty(p - 1, dtype=float)
    r1 = empty(p - 1, dtype=float)
    v1 = zeros(p,     dtype=float)
    
    u   = zeros(3,    dtype=float)
    up  = zeros(3,    dtype=float)
    uxb = zeros(3,    dtype=float)
    tmp = zeros(3,    dtype=float)
    E   = zeros(3,    dtype=float)
    B   = zeros(3,    dtype=float)
    S   = zeros(3,    dtype=float)

    s0 = 0
    s1 = 0

    qprime = dt*qe/(2*me)
    npart = len(particles)
    dz = Lz/Nb

    #$ omp parallel
    #$ omp do private ( ip, pos, s0, l0, r0, v0, s1, l1, r1, v1, E, B, normB, r, S, u, uxb, up, tmp, il0, i0, ii0, bi0, il1, i1, ii1, bi1)
    for ip in range(0, npart):
        # ... field interpolation
        E[:] = 0.
        B[:] = 0.
        B[2] = B0z

        pos = particles[ip, 0]
        
        s0 = find_span(T, p, pos)
        s1 = find_span(tz, p - 1, pos)
        
        basis_funs(T, p, pos, s0, l0, r0, v0)
        basis_funs(tz, p - 1, pos, s1, l1, r1, v1)

        for il0 in range(0, p + 1):

            i0 = s0 - il0
            ii0 = i0%Nb
            bi0 = v0[p - il0]

            E[0] += ex[ii0]*bi0
            E[1] += ey[ii0]*bi0
            
        for il1 in range(0, p):

            i1 = s1 - il1
            ii1 = i1%Nb
            bi1 = v1[p - 1 - il1]/dz

            B[0] += bx[ii1]*bi1
            B[1] += by[ii1]*bi1
        # ...

        u = particles[ip, 1:4] + qprime*E[:]
        gamma = (1 + (u[0]**2 + u[1]**2 + u[2]**2)/c**2)**(1/2)
        cross(u, B[:], uxb)
        uxb[:] = qprime*uxb[:]/gamma
        normB = B[0]**2 + B[1]**2 + B[2]**2
        r = 1 + (qprime/gamma)**2*normB
        S[:] = 2*(qprime/gamma)*B[:]/r
        cross(u + uxb, S, tmp)
        up = u + tmp
        particles[ip, 1:4] = up + qprime*E[:]
        gamma = (1 + (particles[ip, 1]**2 + particles[ip, 2]**2 + particles[ip, 3]**2)/c**2)**(1/2)
        particles[ip, 0] += dt*particles[ip, 3]/gamma
        particles[ip, 0] = particles[ip, 0]%Lz 

    #$ omp end do
    #$ omp end parallel

    # TODO remove this line later, once fixed in pyccel
    ierr = 0


#==============================================================================
@external_call
@types('double[:,:](order=F)','double','double','double','double[:]','double[:]','int','double[:]','double[:]','double[:]','double[:]','double','double','double', 'double')
def borisGemRel_bc_2(particles, dt, qe, me, T, tz, p, ex, ey, bx, by, B0z, xi, Lz, c):

    from numpy      import empty
    from numpy      import zeros

    l0 = empty(p,     dtype=float)
    r0 = empty(p,     dtype=float)
    v0 = zeros(p + 1, dtype=float)
    
    l1 = empty(p - 1, dtype=float)
    r1 = empty(p - 1, dtype=float)
    v1 = zeros(p,     dtype=float)
    
    u   = zeros(3,    dtype=float)
    up  = zeros(3,    dtype=float)
    uxb = zeros(3,    dtype=float)
    tmp = zeros(3,    dtype=float)
    E   = zeros(3,    dtype=float)
    B   = zeros(3,    dtype=float)
    S   = zeros(3,    dtype=float)
    ez  = zeros(3,    dtype=float)
    rho = zeros(3,    dtype=float)
    
    ez[2] = 1.

    s0 = 0
    s1 = 0

    qprime = dt*qe/(2*me)
    npart = len(particles)

    #$ omp parallel
    #$ omp do private ( ipart, pos, s0, l0, r0, v0, s1, l1, r1, v1, E, B, rho, normB, r, S, u, uxb, up, tmp, il0, i0, bi0, il1, i1, bi1 )
    for ipart in range(0, npart):
        # ... field interpolation (wave + background)
        pos = particles[ipart, 0]
        
        s0 = find_span(T, p, pos)
        s1 = find_span(tz, p - 1, pos)
        
        basis_funs(T, p, pos, s0, l0, r0, v0)
        basis_funs(tz, p - 1, pos, s1, l1, r1, v1)
        
        if (particles[ipart, 0] + dt*particles[ipart, 3] > Lz) or (particles[ipart, 0] + dt*particles[ipart, 3] < 0.):
            particles[ipart, 3] = -particles[ipart, 3]
            
        E[:] = 0.
        B[:] = 0.
        B[2] = B0z*(1 + xi*(pos - Lz/2)**2)
        
        rho[:] = 0.

        for il0 in range(0, p + 1):

            i0 = s0 - il0
            bi0 = v0[p - il0]

            E[0] += ex[i0]*bi0
            E[1] += ey[i0]*bi0
            
        for il1 in range(0, p):

            i1 = s1 - il1
            bi1 = v1[p - 1 - il1]*p/(tz[i1 + p] - tz[i1])

            B[0] += bx[i1]*bi1
            B[1] += by[i1]*bi1
            
        cross(particles[ipart, 1:4], ez, rho)
        B[0] += me*rho[0]/(qe*B[2])*(pos - Lz/2)*B0z*xi
        B[1] += me*rho[1]/(qe*B[2])*(pos - Lz/2)*B0z*xi
        # ...

        u = particles[ipart, 1:4] + qprime*E[:]
        gamma = (1 + (u[0]**2 + u[1]**2 + u[2]**2)/c**2)**(1/2)
        cross(u, B[:], uxb)
        uxb[:] = qprime*uxb[:]/gamma
        normB = B[0]**2 + B[1]**2 + B[2]**2
        r = 1 + (qprime/gamma)**2*normB
        S[:] = 2*(qprime/gamma)*B[:]/r
        cross(u + uxb, S, tmp)
        up = u + tmp
        particles[ipart, 1:4] = up + qprime*E[:]
        gamma = (1 + (particles[ipart, 1]**2 + particles[ipart, 2]**2 + particles[ipart, 3]**2)/c**2)**(1/2)
        particles[ipart, 0] += dt*particles[ipart, 3]/gamma

    #$ omp end do
    #$ omp end parallel

    # TODO remove this line later, once fixed in pyccel
    ierr = 0