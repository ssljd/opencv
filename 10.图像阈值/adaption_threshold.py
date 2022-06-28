import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../img/IMG_20201016_212915.jpg',0)
# 中值滤波
img = cv2.medianBlur(img, 5)
ret, th1 = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)      # 阈值取自相邻区域的平均值
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 阈值取值相邻区域的加权和，权重为一个高斯窗口
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()