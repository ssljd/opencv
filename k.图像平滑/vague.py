#图像模糊(图像平滑）：平均,高斯模糊，中值模糊，双边滤波
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('../img/1.png')
blur = cv2.blur(img,(5,5))#平均
blur1 = cv2.GaussianBlur(img,(5,5),0)#高斯模糊
median = cv2.medianBlur(img,5)#中值模糊
blur2 = cv2.bilateralFilter(img,9,75,75)#双边滤波
# dst = cv2.filter2D(img,-1,blur)
plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(232),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]),plt.yticks([])
plt.subplot(233),plt.imshow(blur1),plt.title('Gaussian')
plt.xticks([]),plt.yticks([])
plt.subplot(234),plt.imshow(img),plt.title('Median')
plt.xticks([]),plt.yticks([])
plt.subplot(235),plt.imshow(img),plt.title('Bilateral')
plt.xticks([]),plt.yticks([])
plt.show()