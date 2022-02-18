import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import pi
import scipy.signal as sgn

import SNR_lib as SNR
import BW_PSF_lib as BW
import FRC_lib as FRC

#%% image simulation

wl = 0.488
k = 2*pi/wl
n = 1.33
Z = 0
px = 0.1 # pixel size in um
NA = 0.8

Theo_res = 0.61*wl*n/NA
print('Theoretical Resolution =', Theo_res, 'um')

N = 256
M = 256

c = [N//2, M//2]

x = np.arange(N)
y = np.arange(M)

X, Y = np.meshgrid(x,y)

R = np.sqrt( (X-c[0])**2 + (Y-c[1])**2 )*px

psf = 5e3*BW.PSF(Z, R, NA, wl, n)
signal = np.max(psf)

P = rnd.randint(low=0, high=1001, size=(N,M)) // 1000

Z = sgn.fftconvolve(P, psf,'same')

bkg = 10
std = 50

img = []

img.append( Z + bkg + rnd.normal(0, std, np.shape(Z)) )
img.append( Z + bkg + rnd.normal(0, std, np.shape(Z)) )

plt.figure()
plt.imshow(img[0])
plt.figure()
plt.imshow(img[1])
    
#%% SNR analysis

signal, noise, snr, res_um = SNR.SNR_analysis(img, px)