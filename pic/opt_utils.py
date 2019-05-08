from pyccel.decorators import types
from pyccel.decorators import pure
from pyccel.decorators import external_call

#==============================================================================
@pure
@types('double[:]','double[:]','double[:]')
def cross( a, b, r ):
    r[0] = a[1] * b[2] - a[2] * b[1]
    r[1] = a[2] * b[0] - a[0] * b[2]
    r[2] = a[0] * b[1] - a[1] * b[0]

#==============================================================================
@pure
@types('double[:]','int','double')
def find_span( knots, degree, x ):
    """
    Determine the knot span index at location x, given the
    B-Splines' knot sequence and polynomial degree. See
    Algorithm A2.1 in [1].

    For a degree p, the knot span index i identifies the
    indices [i-p:i] of all p+1 non-zero basis functions at a
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
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
               high = span
            else:
               low  = span
            span = (low+high)//2
        returnVal = span

    return returnVal

#==============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs( knots, degree, x, span, left, right, values ):
    """
    Compute the non-vanishing B-splines at location x,
    given the knot sequence, polynomial degree and knot
    span. See Algorithm A2.2 in [1].

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

    Notes
    -----
    The original Algorithm A2.2 in The NURBS Book [1] is here
    slightly improved by using 'left' and 'right' temporary
    arrays that are one element shorter.

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
    for j in range(0,p):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved


#==============================================================================

#==============================================================================
@external_call
@types('double[:]','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','double[:,:](order=F)')
def fieldInterpolation_bc_1(particles_pos, knots, p, Nb, ex, ey, bx, by, Ep, Bp):

    # ... needed for splines evaluation
    from numpy      import empty
    from numpy      import zeros

    left   = empty( p  , dtype=float )
    right  = empty( p  , dtype=float )
    values = zeros( p+1, dtype=float )

    span = 0
    # ...

    # ...
    npart = len(particles_pos)
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, left, right, values )

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            Ep[ipart, 0] += ex[ii]*bi
            Ep[ipart, 1] += ey[ii]*bi
            Bp[ipart, 0] += bx[ii]*bi
            Bp[ipart, 1] += by[ii]*bi
    # ...


#==============================================================================
@external_call
@types('double[:,:](order=F)','double[:]','double[:]','double[:]','int','int','double','double','double[:]')
def hotCurrent_bc_1(particles_vel, particles_pos, particles_wk, knots, p, Nb, qe, c, jh):

    # ... needed for splines evaluation
    from numpy      import empty
    from numpy      import zeros

    left   = empty( p  , dtype=float )
    right  = empty( p  , dtype=float )
    values = zeros( p+1, dtype=float )

    span = 0
    # ...

    npart = len(particles_pos)

    #$ omp parallel
    #$ omp do reduction ( + : jh ) private ( ipart, pos, span, left, right, values, wk, vx, vy, il, i, ii, bi )
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, left, right, values )

        wk = particles_wk[ipart]
        vx = particles_vel[ipart, 0]
        vy = particles_vel[ipart, 1]

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            jh[2*ii] += vx * wk * bi
            jh[2*ii + 1] += vy * wk * bi

    #$ omp end do
    #$ omp end parallel

    jh = qe/npart*jh

#==============================================================================
@external_call
@types('double[:,:](order=F)','double','double','double','double','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','double')
def borisPush_bc_1(particles, dt, qe, me, Lz, knots, p, Nb, ex, ey, bx, by, B0z):

    from numpy      import empty
    from numpy      import zeros

    left   = empty( p  , dtype=float )
    right  = empty( p  , dtype=float )
    values = zeros( p+1, dtype=float )
    u      = zeros( 3  , dtype=float )
    uprime = zeros( 3  , dtype=float )
    uxb    = zeros( 3  , dtype=float )
    tmp    = zeros( 3  , dtype=float )
    E      = zeros( 3  , dtype=float )
    B      = zeros( 3  , dtype=float )
    S      = zeros( 3  , dtype=float )

    span = 0

    qprime = dt*qe/(2*me)
    npart = len(particles)

    #$ omp parallel
    #$ omp do private ( ipart, pos, span, left, right, values, E, B, normB, r, S, u, uxb, uprime, tmp, il, i, ii, bi )
    for ipart in range(0, npart):
        # ... field interpolation
        E[:] = 0.
        B[:] = 0.
        B[2] = B0z

        pos = particles[ipart,0]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, left, right, values )

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            E[0] += ex[ii]*bi
            E[1] += ey[ii]*bi
            B[0] += bx[ii]*bi
            B[1] += by[ii]*bi
        # ...

        normB = B[0]**2 + B[1]**2 + B[2]**2
        r = 1 + (qprime**2)*normB
        S[:] = 2*qprime*B[:]/r
        u = particles[ipart, 1:4] + qprime*E[:]
        cross( u, B[:], uxb )
        uxb[:] = qprime*uxb[:]
        cross( u + uxb, S, tmp )
        uprime = u + tmp
        particles[ipart, 1:4] = uprime + qprime*E[:]

    #$ omp end do
    #$ omp end parallel

    # TODO remove this line later, once fixed in pyccel
    ierr = 0



