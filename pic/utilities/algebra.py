from pyccel.decorators import types
from pyccel.decorators import pure

#==============================================================================
@pure
@types('double[:]','double[:]','double[:]')
def cross( a, b, r ):
    r[0] = a[1] * b[2] - a[2] * b[1]
    r[1] = a[2] * b[0] - a[0] * b[2]
    r[2] = a[0] * b[1] - a[1] * b[0]
