import cv2
import numpy as np
from matplotlib import pyplot as plt

#roi是我们需要找到的对象或对象的区域
roi = cv2.imread('../img/1.png')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#目标是我们搜索的图像
target = cv2.imread('../img/2.png')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
#使用calcHist查找直方图。也可以用np.historogram2d完成
M = cv2.calcHist([hsv],[0,1],None,[200,256],[0,200,0,256])
I = cv2.calcHist([hsvt],[0,1],None,[200,256],[0,200,0,256])
R = M/I
# print(R)
h,s,v = cv2.split(hsvt)
B = R[h.ravel(),s.ravel()]
print(B)
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])
print(B)
#使用圆盘算子做卷积
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
B = cv2.filter2D(B,-1,disc)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
ret,thresh = cv2.threshold(B,50,255,0)
plt.subplot(111),plt.imshow(thresh,'gray')
plt.title("B")
plt.show()