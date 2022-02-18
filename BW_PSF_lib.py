import numpy as np
from scipy import pi
import scipy.integrate as integrate
import scipy.special as special
# import quadpy
#import time

def integrand(rho, R, Z, NA, wl, n):
    # start_time = time.time()
    k = 2*pi/wl
    
    R = np.repeat(R[:, :, np.newaxis], len(rho), axis=2)
    rho = np.repeat(rho[np.newaxis, :], np.shape(R)[0], axis=0)
    rho = np.repeat(rho[:, np.newaxis, :], np.shape(R)[1], axis=1)
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    amp = special.jv(0, k*NA*R*rho)
    phase = k*rho**2*Z*(NA**2)/n
    # print('length of rho:', np.shape(rho))
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    return rho * amp * np.exp(-0.5j*phase)

# def PSF(R, Z, NA, wl, n):
#     N, M = np.shape(R)
#     result = np.empty([N, M], dtype=np.complex128)
#     for i in range(N):
#         for j in range(M):
#             result[i, j] = quadpy.quad( lambda x: intern(x, R[i, j], Z, NA, wl, n), 0, 1 )[0]
#     return np.abs(result)**2

# def PSF(R, Z, NA, wl, n):
#     result = quadpy.quad( lambda x: intern(x, R, Z, NA, wl, n), 0, 1, epsabs = 1e-4, epsrel = 1e-4 )[0]
#     return np.abs(result)**2

def PSF(Z, R, NA, wl, n):
    x = np.linspace(0, 1, endpoint=True, num=200)
    result = integrate.simps(integrand(x, R, Z, NA, wl, n), x)
    return np.abs(result)**2