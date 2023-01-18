import cv2
import numpy as np

# read image as grayscale
img = cv2.imread('mot.jpg',0)

# take dft
dft = np.fft.fft2(img)

# get power spectral density of dft = square of magnitude
# where abs of complex number is the magnitude
pspec = (np.abs(dft))**2
print(np.amin(pspec))
print(np.amax(pspec))

# estimate noise power spectral density
# try different values to achieve compromise between noise reduction and softening/blurring
#noise = 100000000
noise = 500000000
#noise = 1000000000
#noise = 5000000000

# do wiener filtering
wiener = pspec/(pspec+noise)
wiener = wiener*dft

# do dft to restore
restored = np.fft.ifft2(wiener)

# take real() component (or do abs())
restored = np.real(restored)
print(np.amin(restored))
print(np.amax(restored))

# clip and convert to uint8
restored = restored.clip(0,255).astype(np.uint8)

# save results
#cv2.imwrite('pandas_noisy_restored_100000000.jpg',restored)
#cv2.imwrite('pandas_noisy_restored_500000000.jpg',restored)
#cv2.imwrite('pandas_noisy_restored_1000000000.jpg',restored)
cv2.imwrite('pandas_noisy_restored_5000000000.jpg',restored)

# display results
cv2.imshow("input", img)
cv2.imshow("restored", restored)
cv2.waitKey(0)