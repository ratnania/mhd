import numpy as np

from bsplines import find_span, basis_funs


def block_borisPush_bc_1(particles, dt, B, E, qe, me, Lz):
    from numpy.linalg import norm

    qprime = dt*qe/(2*me)
    H = qprime*B
    r = 1 + norm(H, axis = 1)**2
    npart = len(particles)
    for ipart in range(0, npart):
        S = 2*H[ipart, :]/r[ipart]
        x0 = particles[ipart, 0]
        u = particles[ipart, 1:4] + qprime*E[ipart, :]
        uprime = u + np.cross(u + np.cross(u, H[ipart, :]), S)
        particles[ipart, 1:4] = uprime + qprime*E[ipart, :]

def borisPush_bc_1(particles, dt, B, E, qe, me, Lz):
    chunksize = 10
    npart = len(particles)
    nsize = npart // chunksize
    for i in range(0, nsize):
        ib = i*chunksize
        ie = ib + chunksize
        pos = particles[ib:ie, :]
        Bs = B[ib:ie, :]
        Es = E[ib:ie, :]
        block_borisPush_bc_1(pos, dt, Bs, Es, qe, me, Lz)


def block_fieldInterpolation_bc_1(particles_pos, knots, p, Nb, ex, ey, bx, by, values):
    Ep = np.zeros((len(particles_pos), 2))
    Bp = np.zeros((len(particles_pos), 2))

    # ...
    npart = len(particles_pos)
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, values )

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            Ep[ipart, 0] += ex[ii]*bi
            Ep[ipart, 1] += ey[ii]*bi
            Bp[ipart, 0] += bx[ii]*bi
            Bp[ipart, 1] += by[ii]*bi
    # ...

    return Ep, Bp

def fieldInterpolation_bc_1(particles_pos, knots, p, Nb, ex, ey, bx, by, values):
    chunksize = 10

    npart = len(particles_pos)

    Ep = np.zeros((npart, 2))
    Bp = np.zeros((npart, 2))

    nsize = npart // chunksize
    for i in range(0, nsize):
        ib = i*chunksize
        ie = ib + chunksize
        pos = particles_pos[ib:ie]
        Ep[ib:ie,:], Bp[ib:ie,:] = block_fieldInterpolation_bc_1(pos, knots, p, Nb, ex, ey, bx, by, values)

    return Ep, Bp

def block_hotCurrent_bc_1(particles_vel, particles_pos, particles_wk, knots, p, Nb, values, jh):

    npart = len(particles_pos)
    for ipart in range(0, npart):
        pos = particles_pos[ipart]
        span = find_span( knots, p, pos )
        basis_funs( knots, p, pos, span, values )

        wk = particles_wk[ipart]
        vx = particles_vel[ipart, 0]
        vy = particles_vel[ipart, 1]

        for il in range(0, p + 1):

            i = span - il
            ii = i%Nb
            bi = values[p-il]

            jh[2*ii] += vx * wk * bi
            jh[2*ii + 1] += vy * wk * bi

def hotCurrent_bc_1(particles_vel, particles_pos, particles_wk, knots, p, Nb, qe, c, values):

    chunksize = 10

    jh = np.zeros(2*Nb)

    npart = len(particles_pos)

    nsize = npart // chunksize
    for i in range(0, nsize):
        ib = i*chunksize
        ie = ib + chunksize

        vel = particles_vel[ib:ie,:]
        pos = particles_pos[ib:ie]
        wk  = particles_wk[ib:ie]
        block_hotCurrent_bc_1(vel, pos, wk, knots, p, Nb, values, jh)

    jh = qe/npart*jh
    return jh
