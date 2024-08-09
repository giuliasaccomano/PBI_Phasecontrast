import tifffile as tiff
import os, re
import scipy.constants as constant
import numpy as np
from pyfftw.interfaces.numpy_fft import rfft2, irfft2, fftshift

Bronn_dir = 'D:\\folders_test\\test_Bronnikov\\'
indir = Bronn_dir + 'dataset\\'

def split_and_sort(value):
    """ 
    Returns the element in the list or array "value" sorted by their numerical order 
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def Paganin(rad, dist, delta, beta, energy, pixsize):        
    rad_freq = rfft2(rad) 
    '''from Paganin et al., 2002'''
    perm_in_vacuum = constant.physical_constants['vacuum mag. permeability']
    #  lambda
    wlen =  perm_in_vacuum[0] / energy
    mu = 4*np.pi*beta / wlen
    # Set the transformed frequencies according to pixelsize:
    rows = rad.shape[0]
    cols = rad.shape[1]
    ulim = np.arange(-(cols) / 2, (cols) / 2)
    ulim = ulim * (2 * np.pi / (cols * pixsize))
    vlim = np.arange(-(rows) / 2, (rows) / 2)
    vlim = vlim * (2 * np.pi / (rows * pixsize))
    u,v = np.meshgrid(ulim, vlim)
    filtre =  1 + (dist*delta*(u**2+v**2)) / mu
    den = fftshift(filtre)
    den = den[:,0:int(den.shape[1] / 2) + 1] 
    rad_freq = rad_freq / den
    im = irfft2( rad_freq)
    im = im.astype(np.float32)
    im = -1 / mu * np.log(im)
    return im 

from scipy import ndimage

def Bronnikov_AC(im_phrt, k, dist, energy):      
    perm_in_vacuum = constant.physical_constants['vacuum mag. permeability']
    wlen =  perm_in_vacuum[0] / energy      
    gamma = k*(wlen*dist/(2*np.pi*1))
    Corr = 1 - gamma*np.abs(ndimage.laplace(im_phrt))
    return Corr, gamma

listdir = os.listdir(indir)

tomolist = []
flatlist = []
darklist = []
for i in listdir:
    splits = split_and_sort(i)
    if splits[0] == 'tomo_':
        tomolist.append(i)
    elif splits[0] == 'flat_':
        flatlist.append(i)
    elif splits[0] == 'dark_':
        darklist.append(i)

im = tiff.imread(os.path.join(indir, tomolist[0]))
# Extract shape
dim = im.shape
fx = dim[0]
fy = dim[1]

# Provide phase retrieval information
energy = 19435.6377/1000
distance = 250
delta = 1E-07
beta = 1E-09
pixsize = 0.002

# Set k factor
k = 1/(2.57e-06)
# Extract gamma
im_phrt = Paganin(im,distance,delta,beta,energy,pixsize)
im_bronn, gamma = Bronnikov_AC(im_phrt, k, distance, energy)

# Create output folder
outdir  = Bronn_dir + 'res_gamma' + str(gamma)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Perform flat 
n_flat = len(flatlist)
flat_avg = np.zeros([dim[0], dim[1]])
for i in flatlist:
    flat = tiff.imread(os.path.join(indir, i))
    flat_avg = flat_avg + flat
flat_avg = flat_avg / n_flat

# Perform dark
n_dark = len(darklist)
dark_avg = np.zeros([dim[0], dim[1]])
for i in flatlist:
    dark = tiff.imread(os.path.join(indir, i))
    dark_avg = dark_avg + dark
dark_avg = dark_avg / n_dark

flat_dark = flat_avg - dark_avg

# Perform Paganin + Bronnikov
for i in tomolist:
    print(i)
    im = tiff.imread(os.path.join(indir, i))
    #im = np.divide((im - dark_avg), flat_dark)
    im_phrt = Paganin(im,distance,delta,beta,energy,pixsize)
    tiff.imsave(os.path.join(outdir, i), im_phrt)
    im_bronn, _ = Bronnikov_AC(im_phrt, k, distance, energy)
    tiff.imsave(os.path.join(outdir, 'phrt'), im_bronn)
    im_out = im / im_bronn
    tiff.imsave(os.path.join(outdir, 'bronn'), im_out)


