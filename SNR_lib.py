import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import pi
import scipy.signal as sgn

import FRC_lib as FRC

def SNR_analysis(img, px = 1):
    
    # FRC calculation
    
    N, M = np.shape(img[0])

    frc = FRC.FRC(img[0], img[1])
    F = len(frc)

    max_kpx = 1 / np.sqrt(2)

    kpx = np.linspace(0, max_kpx, F, endpoint = True)
    k = kpx / px

    frc_smooth = FRC.smooth(k, frc)
    
    # Resolution measurement
    
    th, idx = FRC.nsigma_threshold(frc_smooth, img[0], 3)

    res_um = (1/k[idx])
    
    # Image filtering
    
    img_lp = FRC.sigmoidFilterLP(img[0], idx, 0.02)
    signal = np.mean(img_lp)
    
    img_hp = FRC.sigmoidFilterHP(img[0], idx, 0.02)
    noise = np.std(img_hp)
    
    snr = signal/noise
    
    return signal, noise, snr, res_um