# ... creates a periodic B-spline basis of degree p with equally spaced elements and the corresponding quadrature grid and weights

from Utils.bsplines import Bspline
import numpy as np

def createBasis(Lz,Nz,p):

    dz = Lz/Nz
    zj = np.linspace(0,Lz,Nz+1)

    left = np.linspace(-p*dz,-dz,p)
    right = np.linspace(Lz+dz,Lz+p*dz,p)
    Tbsp = np.array(list(left)  + list(zj) + list(right))
    bsp = Bspline(Tbsp,p)
    N = len(Tbsp) - p - 1
    xi,wi = np.polynomial.legendre.leggauss(p+1)
    quad_points = np.zeros((p+1)*Nz)
    weights = np.zeros((p+1)*Nz)

    for i in range(0,Nz):
        a1 = zj[i]
        a2 = zj[i+1]
        xis = (a2-a1)/2*xi + (a1+a2)/2
        quad_points[(p+1)*i:(p+1)*i+(p+1)] = xis
        wis = (a2-a1)/2*wi
        weights[(p+1)*i:(p+1)*i+(p+1)] = wis
	
    return bsp,N,quad_points,weights
