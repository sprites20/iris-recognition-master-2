"""
=====================
Image Deconvolution
=====================
In this example, we deconvolve an image using Richardson-Lucy
deconvolution algorithm ([1]_, [2]_).

The algorithm is based on a PSF (Point Spread Function),
where PSF is described as the impulse response of the
optical system. The blurred image is sharpened through a number of
iterations, which needs to be hand-tuned.

.. [1] William Hadley Richardson, "Bayesian-Based Iterative
       Method of Image Restoration",
       J. Opt. Soc. Am. A 27, 1593-1607 (1972), :DOI:`10.1364/JOSA.62.000055`

.. [2] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2
from scipy.signal import wiener
from scipy.misc import face

from skimage import color, data, restoration

import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian
import matplotlib.pyplot as plt

from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
import random
import cv2

import glob
from pathlib import Path

'''
#Wiener
rng = np.random.default_rng()

image = color.rgb2gray(data.astronaut())

psf = np.ones((5, 5)) / 25
image = conv2(image, psf, 'same')
# Add Noise to Image
image_noisy = image.copy()
image_noisy += (rng.poisson(lam=25, size=image.shape) - 10) / 255.

deconvolved = restoration.wiener(image_noisy, psf, 1, clip=False)
#print deconvolved
plt.imshow(image_noisy, cmap='gray')
plt.show()
plt.imshow(deconvolved, cmap='gray')
plt.show()  

'''
#Anisotropic
import numpy as np
import warnings
 
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
    """
    Anisotropic diffusion.
 
    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)
 
    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration
 
    Returns:
            imgout   - diffused image.
 
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
 
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
 
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes
 
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
 
    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.
 
    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
 
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
 
    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
 
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)
 
    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
 
    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep
 
        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
 
        fig.canvas.draw()
 
    for ii in range(niter):
 
        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
 
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
 
        # update the image
        imgout += gamma*(NS+EW)
 
        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)
 
    return imgout

def lucyrichardson(imgpath):
    """
    #LR
    rng = np.random.default_rng()
    img = cv2.imread('Original.jpg')
    astro = color.rgb2gray(img)
    astro = conv2(astro, psf, 'same')

    # Add Noise to Image
    astro_noisy = astro.copy()
    astro_noisy += (rng.poisson(lam=25, size=astro.shape) - 10) / 255.

    astro_noisy = cv2.imread('MotionBlur_001_1_1.jpg')
    astro_noisy = color.rgb2gray(astro_noisy)
    #imgout = anisodiff(astro_noisy, niter=4)
    # Restore Image using Richardson-Lucy algorithm
    """
    psf = np.ones((5, 5)) / 25
    astro_noisy = cv2.imread( imgpath)
    astro_noisy = color.rgb2gray(astro_noisy)
    astro_noisy = anisodiff(astro_noisy, niter=4)
    deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf)
    
    #deconvolved_RL = cv2.cvtColor(deconvolved_RL, cv2.COLOR_GRAY2RGB)
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
       a.axis('off')
       
    ax[0].imshow(img)
    ax[0].set_title('Original Data')

    ax[1].imshow(astro_noisy)
    ax[1].set_title('Noisy data')

    #cv2.imwrite("DCd.jpg", deconvolved_RL)
    ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
    #ax[2].save('DCd.jpg')
    #fig.savefig('DCd.jpg')
    ax[2].set_title('Restoration using\nRichardson-Lucy')
    fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
    plt.show()
    """
    
    thisimg = deconvolved_RL*255
    
    height, width = thisimg.shape
    print(height, width)
    
    p1x, p1y = int(width * .075), int(height * .075)
    p2x, p2y = int(width - p1x), int(height - p1y)
    
    print(p1x, p1y)
    print(p2x, p2y)
    croppedimg = thisimg[p1y:p2y, p1x:p2x]
    
    somepath = imgpath.split('\\')[-1]
    #cv2.imshow("test.jpg", croppedimg)
    cv2.imwrite("Anisotropic_LucyRichardson_" + somepath, croppedimg)

#lucyrichardson('MotionBlur_001_1_1.jpg') , "poisson", "gaussianblur", "median", "bilateral", "motion", "gaussianblur-gaussian", "gaussianblur-speckle", "gaussianblur-saltandpepper", "gaussianblur-poisson", "median-gaussian", "median-speckle", "median-saltandpepper", "median-poisson", "bilateral-gaussian", "bilateral-speckle", "bilateral-saltandpepper", "bilateral-poisson", "motion-gaussian", "motion-speckle", "motion-saltandpepper", "motion-poisson"]
datas = ["gaussian", "speckle", "saltandpepper"]
for i in datas:
    path = "./" + i + "/**"
    globity = glob.glob(path, recursive=True)
    for path in globity:
        try:
            if path.endswith('jpg'):
                #print(path)
                somepath = path.split('\\')[-1]
                print(somepath)
                lucyrichardson(path)
        except Exception as e:
            print(e)

    