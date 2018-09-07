# This function computes the complex frequency w for a fixed wavenumber k of the electron hybrid model using the Newton method (R-waves: pol = +1, L-waves: pol = -1)

import numpy as np
import scipy.special as sp


def solveDispersionHybrid(k,pol,c,wce,wpe,wpar,wperp,nuh,initial_guess,tol,max_it = 100):
	
	Taniso = 1 - wperp**2/wpar**2

	
	def Z(xi):
		return np.sqrt(np.pi)*np.exp(-xi**2)*(1j - sp.erfi(xi))

	def Zprime(xi):
    		return -2*(1 + xi*Z(xi))


	def Dhybrid(k,w,pol):
		xi = (w + pol*wce)/(k*np.sqrt(2)*wpar)

		return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce)) + nuh*wpe**2/w**2*(w/(k*np.sqrt(2)*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi)))	

   
	def Dhybridprime(k,w,pol):
		xi = (w + pol*wce)/(k*np.sqrt(2)*wpar)
		xip = 1/(k*np.sqrt(2)*wpar)

		return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2) - 2*nuh*wpe**2/w**3*(w/(np.sqrt(2)*k*wpar)*Z(xi) - Taniso*(1 + xi*Z(xi))) + nuh*wpe**2/w**2*(1/(np.sqrt(2)*k*wpar)*Z(xi) + w/(np.sqrt(2)*k*wpar)*Zprime(xi)*xip - Taniso*(xip*Z(xi) + xi*Zprime(xi)*xip)) 

	w = initial_guess
	counter = 0
	
	while True:
		wnew = w - Dhybrid(k,w,pol)/Dhybridprime(k,w,pol)

		if np.abs(wnew - w) < tol or counter == max_it:
			w = wnew
			break

		w = wnew
		counter += 1

	return w,counter


def solveDispersionHybridExplicit(k,pol,c,wce,wpe,wpar,wperp,nuh,initial_guess,tol,max_it = 100):

	def Dcold(k,w,pol):
		return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce))

	def Dcoldprime(k,w,pol):
		return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2)

	wr = initial_guess
	counter = 0

	while True:
		wnew = wr - Dcold(k,wr,pol)/Dcoldprime(k,wr,pol)

		if np.abs(wnew - wr) < tol or counter == max_it:
			wr = wnew
			break

		wr = wnew
		counter += 1

	vR = (wr + pol*wce)/k

	wi = 1/(2*wr - pol*wpe**2*wce/(wr + pol*wce)**2)*np.sqrt(2*np.pi)*wpe**2*nuh*vR/wpar*np.exp(-vR**2/(2*wpar**2))*(wr/(2*(-pol*wce - wr)) + 1/2*(1 - wperp**2/wpar**2))

	return wr,wi,counter




def solveDispersionCold(k,pol,c,wce,wpe,initial_guess,tol,max_it = 100):

	def Dcold(k,w,pol):
		return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce))

	def Dcoldprime(k,w,pol):
		return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2)

	wr = initial_guess
	counter = 0

	while True:
		wnew = wr - Dcold(k,wr,pol)/Dcoldprime(k,wr,pol)

		if np.abs(wnew - wr) < tol or counter == max_it:
			wr = wnew
			break

		wr = wnew
		counter += 1

	return wr,counter


def solveDispersionArtificial(k,pol,c,wce,wpe,c1,c2,mu0,initial_guess,tol,max_it = 100):

	def Dart(k,w,pol):
		return 1 - k**2*c**2/w**2 - wpe**2/(w*(w + pol*wce)) + 1/w*(1j*c1 - pol*c2)

	def Dartprime(k,w,pol):
		return 2*k**2/w**3 + wpe**2*(2*w + pol*wce)/(w**2*(w + pol*wce)**2) - 1/w**2*(1j*c1 - pol*c2)

	w = initial_guess
	counter = 0

	while True:
		wnew = w - Dart(k,w,pol)/Dartprime(k,w,pol)

		if np.abs(wnew - w) < tol or counter == max_it:
			w = wnew
			break

		w = wnew
		counter += 1

	return w,counter
