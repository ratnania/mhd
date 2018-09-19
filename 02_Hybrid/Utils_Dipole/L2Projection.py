# Computes the L2-projection of a funcion in a given Finite Element basis in the case of periodic (bcs = 1) and Dirichlet b.c. (bcs = 2)

import numpy as np

def L2proj(basis, Nz, p, Lz, quad_points, weights, mass, fun, bcs = 1):

	if bcs == 1:

		f = np.zeros(Nz)

		for ie in range(0, Nz):
			for il in range(0, p+1):

				i = il + ie

				value_f = 0.0

				for g in range(0, p+1):
					gl = ie*(p+1) + g

					value_f += weights[gl]*fun(quad_points[gl])*basis(quad_points[gl],i,0)

				f[i%Nz] += value_f

		fj = np.linalg.solve(mass,f)

		return fj

	elif bcs == 2:

		Ua = fun(0)
		Ub = fun(Lz)
		N = Nz + p

		f = np.zeros(N)
		fj = np.zeros(N)

		for ie in range(0, Nz):
			for il in range(0, p+1):

				i = ie + il

				value_f = 0.0

				for g in range(0, p+1):
					gl = ie*(p + 1) + g

					value_f += weights[gl]*(fun(quad_points[gl])*basis(quad_points[gl],i,0) - Ua*basis(quad_points[gl],0,0)*basis(quad_points[gl],i,0) - Ub*basis(quad_points[gl],N-1,0)*basis(quad_points[gl],i,0))

				f[i] += value_f

		fj[1:N-1] = np.linalg.solve(mass[1:N-1,1:N-1],f[1:N-1])
		fj[0] = Ua
		fj[-1] = Ub 

		return fj



