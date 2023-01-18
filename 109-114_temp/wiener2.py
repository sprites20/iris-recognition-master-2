import os
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

from scipy.ndimage import convolve
from scipy.signal import wiener
from scipy.signal import gaussian
from skimage import color, data, restoration

from scipy.misc import face
from numpy.fft import fft2, ifft2
from PIL import Image

import glob
from pathlib import Path

import numpy as np
from scipy import signal

def wiener_filter(img, kernel_size, noise_power):
    # Compute padding size
    pad_size = (kernel_size - img.shape[0] + 1) // 2

    # Pad the image with zeros
    img_padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)

    # Estimate the power spectrum of the original image
    f = np.fft.fft2(img_padded)
    fshift = np.fft.fftshift(f)
    power_spectrum = np.abs(fshift)**2

    # Create a Gaussian kernel
    x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*noise_power)))

    # Convert the kernel to the frequency domain
    g_f = np.fft.fft2(g)
    fshift_padded = np.pad(fshift, ((0, g_f.shape[0] - fshift.shape[0]), (0, g_f.shape[1] - fshift.shape[1])), mode='constant', constant_values=0)
    # Apply the Wiener filter
    restored_f = g_f / (g_f + noise_power) * fshift_padded
    restored_img = np.fft.ifft2(np.fft.ifftshift(restored_f))

    # Crop the filtered image to remove the padding
    filtered_img = np.abs(restored_img)[pad_size:pad_size+img.shape[0], pad_size:pad_size+img.shape[1], :]

    return filtered_img

# Test the Wiener filter
#img = np.random.rand(256, 256)  # Generate a random image
#noisy_img = img + np.random.randn(256, 256)  # Add some noise

#filtered_img = wiener_filter(noisy_img, kernel_size=16, noise_power=0.1)

def save_image(image, filename):
    image = image.astype(np.uint8) # Convert image to 8-bit unsigned integer type
    img = Image.fromarray(image) # Create a PIL image object
    img.save(filename) # Save the image


def wiener(imgpath):
    # Load image and convert it to gray scale
    file_name = os.path.join(imgpath)
    img = plt.imread(file_name)
    
    # Create the 2D kernel
    #kernel = gaussian_kernels(5)
    
    # Replicate the 2D kernel along the third dimension to create a 3D kernel
    #kernel = np.stack([kernel, kernel, kernel], axis=-1)
    min_dim = min(img.shape[:2])
    kernel_size = min_dim if min_dim % 2 == 1 else min_dim - 1
    # Apply the Wiener filter
    noise_power=0.1
    filtered_img = wiener_filter(img, kernel_size, noise_power)

    # Save the filtered image
    save_image(filtered_img, "LucyRichardson_" + somepath)
    
datas = ["gaussian", "speckle", "saltandpepper", "poisson", "gaussianblur", "median", "bilateral", "motion", "gaussianblur-gaussian", "gaussianblur-speckle", "gaussianblur-saltandpepper", "gaussianblur-poisson", "median-gaussian", "median-speckle", "median-saltandpepper", "median-poisson", "bilateral-gaussian", "bilateral-speckle", "bilateral-saltandpepper", "bilateral-poisson", "motion-gaussian", "motion-speckle", "motion-saltandpepper", "motion-poisson"]
for i in datas:
    path = "./" + i + "/**"
    globity = glob.glob(path, recursive=True)
    for path in globity:
        #try:
        if path.endswith('jpg'):
            #print(path)
            somepath = path.split('\\')[-1]
            print(somepath)
            wiener(path)
        #except Exception as e:
        #    print(e)