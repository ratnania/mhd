# This function defines the initial conditions for the simulation, i.e. initial Ex,Ey,Bx,By,jx,jy

import numpy as np

def IC(z,ini,amp,k,omega):
    
    if ini == 1:
 
        eps0 = 1.0
        wce = -1.0
        wpe = 2.0

        Ex0 = +amp*np.cos(k*z)
        Ey0 = -amp*np.sin(k*z)
        
        Bx0 = -Ey0*k/omega
        By0 = +Ex0*k/omega
        
        Dj = eps0*wpe**2*(omega - wce)/(wce**2 - omega**2)
        
        jx0 = -Ey0*Dj
        jy0 = +Ex0*Dj
            
        return np.array([Ex0,Ey0,Bx0,By0,jx0,jy0])
    
    elif ini == 2:
        
        Ex0 = +amp*np.real(np.exp(1j*k*z))
        Ey0 = -amp*np.imag(np.exp(1j*k*z))
        
        Bx0 = k*amp*np.imag(1/omega*np.exp(1j*k*z))
        By0 = k*amp*np.real(1/omega*np.exp(1j*k*z))
        
        Dj = eps0*wpe**2*(omega - wce)/(wce**2 - omega**2)
        
        jx0 = amp*np.imag(Dj*np.exp(1j*k*z))
        jy0 = amp*np.real(Dj*np.exp(1j*k*z))
            
        return np.array([Ex0,Ey0,Bx0,By0,Bz0,jx0,jy0])
    
    elif ini == 3:
        
        Ex0 = 0*z
        Ey0 = 0*z
        
        Bx0 = amp*np.sin(k*z)
        By0 = 0*z
        
        
        jx0 = 0*z
        jy0 = 0*z
            
        return np.array([Ex0,Ey0,Bx0,By0,jx0,jy0])
    
    elif ini == 4:
        
        Ex0 = 0*z
        Ey0 = 0*z
        
        Bx0 = amp*np.random.randn()
        By0 = amp*np.random.randn()
        
        jx0 = 0*z
        jy0 = 0*z
            
        return np.array([Ex0,Ey0,Bx0,By0,jx0,jy0])
      
    
    elif ini == 5:
        
        Ex0 = amp*np.random.randn()
        Ey0 = amp*np.random.randn()
        
        Bx0 = amp*np.random.randn()
        By0 = amp*np.random.randn()
        
        jx0 = amp*np.random.randn()
        jy0 = amp*np.random.randn()
        
        return np.array([Ex0,Ey0,Bx0,By0,jx0,jy0])
        
    elif ini == 6:
        
        Ex0 = 0*z
        Ey0 = 0*z
        
        Bx0 = 0*z
        By0 = 0*z
        
        jx0 = 0*z
        jy0 = 0*z
        
        return np.array([Ex0,Ey0,Bx0,By0,jx0,jy0]) 
