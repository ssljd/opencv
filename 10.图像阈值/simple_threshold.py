import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../img/IMG_20201104_180926.jpg',0)
ret1, thresh1 = cv2.threshold(img, 126, 255, cv2.THRESH_BINARY)       # 二值阈值化
ret2, thresh2 = cv2.threshold(img, 126, 255, cv2.THRESH_BINARY_INV)   # 反向二值阈值化并反转
ret3, thresh3 = cv2.threshold(img, 126, 255, cv2.THRESH_TRUNC)        # 截断阈值化
ret4, thresh4 = cv2.threshold(img, 126, 255, cv2.THRESH_TOZERO)       # 超过阈值被置位0
ret5, thresh5 = cv2.threshold(img, 126, 255, cv2.THRESH_TOZERO_INV)   # 低于阈值被置位0
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()