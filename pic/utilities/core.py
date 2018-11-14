from pyccel.decorators import types
from pyccel.decorators import pure

from bsplines import find_span
from bsplines import basis_funs
from algebra  import cross

#==============================================================================
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
@types('double[:,:](order=F)','double','double[:,:](order=F)','double[:,:](order=F)','double','double','double')
def borisPush_bc_1(particles, dt, B, E, qe, me, Lz):

    from numpy      import zeros

    u   = zeros( 3, dtype=float )
    uprime = zeros( 3  , dtype=float )
    uxb = zeros( 3, dtype=float )
    tmp = zeros( 3, dtype=float )
    S      = zeros( 3  , dtype=float )

    qprime = dt*qe/(2*me)
    npart = len(particles)

    for ipart in range(0, npart):
        normB = B[ipart,0]**2 + B[ipart,1]**2 + B[ipart,2]**2
        r = 1 + (qprime**2)*normB
        S = 2*qprime*B[ipart, :]/r
        x0 = particles[ipart, 0]
        u = particles[ipart, 1:4] + qprime*E[ipart, :]
        cross( u, B[ipart, :], uxb )
        uxb = qprime*uxb
        cross( u + uxb, S, tmp )
        uprime = u + tmp
        particles[ipart, 1:4] = uprime + qprime*E[ipart, :]

#==============================================================================
@types('double[:,:](order=F)','double','double','double','double','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','double')
def new_borisPush_bc_1(particles, dt, qe, me, Lz, knots, p, Nb, ex, ey, bx, by, B0z):

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


