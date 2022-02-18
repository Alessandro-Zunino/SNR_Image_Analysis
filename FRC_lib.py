import numpy as np
from scipy import pi
import scipy.signal as sgn
import scipy.fft as ft
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import ndimage

def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = sgn.fftconvolve(image, ar.conj(), mode=mode)
    
    image = sgn.fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(sgn.fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

def drift(corr):
    N = (len(corr) + 1) / 2
    
    idx = np.argwhere(corr==np.max(corr))[0]
    
    nx = np.arange(-N+1, N)
    ny = np.arange(-N+1, N)
    
    dx = nx[idx[0]]
    dy = ny[idx[1]]

    return np.int(dx), np.int(dy)

def driftcorrect(A, B, drift):
    dx, dy = drift
    B = np.roll(B, dy, axis=0)
    B = np.roll(B, dx, axis=1)
    
    if dy>0:
        A = A[dy:, :]
        B = B[dy:, :]
    elif dy<0:
        A = A[:dy, :]
        B = B[:dy, :]
    if dx>0:
        A = A[:, dx:]
        B = B[:, dx:]
    elif dx<0:
        A = A[:, :dx]
        B = B[:, :dx]
        
    return A, B

def smooth(x,y):
    filtered = lowess(y, x, is_sorted=True, frac=0.05, it=0)
    return filtered[:,1]

def hann2d(*args):
    if len(args)>1 :
        N, M = args
    else:
        N, = args
        M=N

    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x,y)
    W = 0.5 * ( 1 - np.cos( (2*pi*X)/(N-1) ) )
    W *= 0.5 * ( 1 - np.cos( (2*pi*Y)/(M-1) ) )
    return W

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    
    tbin = np.bincount(r.ravel(), np.real(data).ravel()).astype(np.complex128)
    tbin += 1j*np.bincount(r.ravel(), np.imag(data).ravel())    
    
    nr = np.bincount(r.ravel())
    radialprofile = tbin
    
    # f = lambda R : data[(r >= R-.5) & (r < R+.5)].mean()
    # max_r = np.int( np.max(r) )
    # R  = np.linspace(0, max_r, num = max_r+1)
    # radialprofile = np.vectorize(f)(R)
    
    return radialprofile, nr

def FRC(im1, im2):
    M, N = np.shape(im1)
    cx = np.int ( ( N + np.mod(N,2) ) / 2)
    cy = np.int ( ( M + np.mod(M,2) ) / 2)
    center = [cx, cy]
    
    im1 = im1 * hann2d(N,M)
    im2 = im2 * hann2d(N,M)
    
    ft1 = ft.fftshift( ft.fft2( im1 ) )
    ft2 = ft.fftshift( ft.fft2( im2 ) )
    
    num = np.real( radial_profile( ft1*np.conj(ft2) , center)[0] )
    den = np.real( radial_profile( np.abs(ft1)**2, center)[0] )
    den = den * np.real( radial_profile( np.abs(ft2)**2, center)[0] )
    den = np.sqrt( den )
    
    # F = len(num)
    # max_kpx = 1 / np.sqrt(2)
    # kpx = np.linspace(0, max_kpx, F, endpoint = True)
    
    # num = num - np.mean ( num[ kpx > 1/res] )
    # den = den - np.mean ( den[ kpx > 1/res] )
    FRC = num / den

    # FC =  np.real( ft1*np.conj(ft2) ) / np.sqrt( abs(ft1)**2 * abs(ft2)**2)
    # FRC = radial_profile( FC, center )
    return FRC

def fixed_threshold(frc, y):
    N = len(frc)
    th = np.ones(N) * y
    try:
        idx = np.argwhere(np.diff(np.sign(frc - th))).flatten()[0]
    except:
        idx = 0
    return th, idx

def nsigma_threshold(frc, img, sigma):
    N, M = np.shape(img)
    cx = np.int ( ( N + np.mod(N,2) ) / 2)
    cy = np.int ( ( M + np.mod(M,2) ) / 2)
    center = [cx, cy]
    
    nr = radial_profile(img, center)[1]
    
    th = sigma/np.sqrt(nr/2)
    try:
        idx1 = np.argwhere(np.diff(np.sign(frc - th))).flatten()
        idx2 = idx1[1]
    except:
        idx2 = 0
    return th, idx2

def sigmoidFilterHP(img, t, s):

    ft1 = ft.fftshift( ft.fft2( img ) )
    
    N, M = np.shape(img)
    cx = np.int ( ( N + np.mod(N,2) ) / 2)
    cy = np.int ( ( M + np.mod(M,2) ) / 2)

    x = np.arange(N)
    y = np.arange(M)
    
    X, Y = np.meshgrid(x,y)
    R = np.sqrt( (X-cx)**2 + (Y-cy)**2 )
    
    S = s*np.min([cx,cy])
    
    sigmoid = 1 / (1 + np.exp( -(R-t)/S ) )
    
    ft2 = ft1 * sigmoid
    
    img_filt = np.real( ft.ifft2 ( ft.ifftshift( ft2 ) ) )
    
    return img_filt

def sigmoidFilterLP(img, t, s):

    ft1 = ft.fftshift( ft.fft2( img ) )
    
    N, M = np.shape(img)
    cx = np.int ( ( N + np.mod(N,2) ) / 2)
    cy = np.int ( ( M + np.mod(M,2) ) / 2)

    x = np.arange(N)
    y = np.arange(M)
    
    X, Y = np.meshgrid(x,y)
    R = np.sqrt( (X-cx)**2 + (Y-cy)**2 )
    
    S = s*np.min([cx,cy])
    
    sigmoid = 1 / (1 + np.exp( (R-t)/S ) )
    
    ft2 = ft1 * sigmoid
    
    img_filt = np.real( ft.ifft2 ( ft.ifftshift( ft2 ) ) )
    
    return img_filt

def FRC_resolution(I1, I2, px, method = '3sigma'):

    frc = FRC(I1, I2)
    F = len(frc)

    max_kpx = 1 / np.sqrt(2)

    kpx = np.linspace(0, max_kpx, F, endpoint = True)
    k = kpx / px

    frc_smooth = smooth(k, frc) # FRC smoothing
    
    if method == 'fixed':
        th, idx = fixed_threshold(frc_smooth, 1/7)
    elif method == '3sigma':
        th, idx = nsigma_threshold(frc_smooth, I1, 3)
    elif method == '5sigma':
        th, idx = nsigma_threshold(frc_smooth, I1, 5)

    res_px = (1/kpx[idx])
    res_um = (1/k[idx])
    
    return res_um, res_px, k, th, frc_smooth, frc

def sub(img, c, r):
    # r has to be less than Nx/2 and Ny/2
    cx, cy = c
    Nx, Ny = np.shape(img)
    
    if (cx - r < 0):
        range_x = range(0, 2*r)
    elif (cx + r > Nx):
        range_x = range(Nx - 2*r, Nx)
    else:
        range_x = range(cx - r, cx + r)

    if (cy - r < 0):
        range_y = range(0, 2*r)
    elif (cy + r > Ny):
        range_y = range(Ny - 2*r, Ny)
    else:
        range_y = range(cy - r, cy + r)
        
    idx = np.ix_(range_x, range_y)
    return img[idx]