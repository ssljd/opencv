import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../img/IMG_20201017_101216f.jpg")
print(img.shape)
rows,cols,ch = img.shape
print(rows)
print(cols)
print(ch)
pts1 = np.float32([[456,365],[3280,365],[368,4126],[3890,4278]])
pts2 = np.float32([[0,0],[2000,0],[0,2000],[2000,2000]])
M = cv2.getPerspectiveTransform(pts1,pts2)#dst是img经过透视变换后得到的图像
dst = cv2.warpPerspective(img,M,(2000,2000))
plt.subplot(131),plt.imshow(img),plt.title('Input')
plt.subplot(133),plt.imshow(dst),plt.title('Output')
plt.show()