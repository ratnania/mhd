# This function computes the complex frequency w for a fixed wavenumber k of the electron hybrid model

import numpy as np
import scipy.special as sp


def solveDispersion(k,pol,wce,wpe,wpar,wperp,nuh,initial_guess,tol):
	
	Taniso = 1 - wperp**2/wpar**2

	
	def Z(xi):
		return np.sqrt(np.pi)*np.exp(-xi**2)*(1j - sp.erfi(xi))

	def Zprime(xi):
    		return -2*(1 + xi*Z(xi))


	def Dhybrid(k,w,pol):
		xi = (w + pol)/(k*np.sqrt(2)*wpar)

		return 1 - k**2/w**2 - wpe**2/(w*(w + pol)) + nuh*wpe**2/w**2*(w/(k*np.sqrt(2)*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi)))	

   
	def Dhybridprime(k,w,pol):
		xi = (w + pol)/(k*np.sqrt(2)*wpar)
		xip = 1/(k*np.sqrt(2)*wpar)

		return 2*k**2/w**3 + wpe**2*(2*w + pol)/(w**2*(w + pol)**2) - 2*nuh*wpe**2/w**3*(w/(np.sqrt(2)*k*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi))) + nuh*wpe**2/w**2*(1/(np.sqrt(2)*k*wpar)*Z(xi) + w/(np.sqrt(2)*k*wpar)*Zprime(xi)*xip - Taniso*(xip*Z(xi) + xi*Zprime(xi)*xip)) 

	wk = initial_guess
	
	while True:
		wnew = wk - Dhybrid(k,wk,pol)/Dhybridprime(k,wk,pol)

		if np.abs(wnew - wk) < tol:
			wk = wnew
			break

		wk = wnew

	return wk 
