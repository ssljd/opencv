import cv2
import numpy as np
from matplotlib import pyplot as plt

# Numpy中的傅里叶变换
img = cv2.imread("../img/IMG_20201017_101216f.jpg",0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# print(fshift)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
rows,cols = img.shape
crow,ccol = rows/2 , cols/2
# print(type(crow))
fshift[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0
f_ishift = np.fft.ifftshift(fshift)  # 逆平移操作
img_back = np.fft.ifft2(f_ishift)    # FFT逆变换
img_back = np.abs(img_back)          # 取绝对值
plt.subplot(221),plt.imshow(img,'gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum,'gray')
plt.title('Magnitude spectrum'),plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.imshow(img_back,'gray')
plt.title('Image after HPF'),plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(img_back)
plt.title('Result in JET'),plt.xticks([]),plt.yticks([])
plt.show()

# OpenCV中的傅里叶变换
img = cv2.imread("../img/IMG_20201017_101216f.jpg",0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
rows,cols = img.shape
crow,ccol = rows/2,cols/2
#首先创建一个遮罩，中心正方形为1，其余为0
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow-30):int(crow+30),int(ccol-30):int(ccol+30)] = 1
#应用掩模和逆DFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0],img_back[:, :, 1])
plt.subplot(131),plt.imshow(img,'gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum,'gray')
plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(img_back,'gray')
plt.title('Image after HPF'),plt.xticks([])
plt.show()