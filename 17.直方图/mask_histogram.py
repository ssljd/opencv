import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../img/IMG_20201017_101216f.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
mask[0:4000,0:4600] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)
#计算带有和不带有掩膜的直方图
#检查掩膜的第三个参数
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.show()