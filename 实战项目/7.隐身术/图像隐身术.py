# ==============导入库=================
import cv2
import numpy as np
# ============读取前景图像、背景图像=============
A = cv2.imread('../img/back.jpg')
cv2.imshow('A', A)
B = cv2.imread('../img/fore.jpg')
cv2.imshow('B', B)
# ===========获取掩膜图像mask1/mask2=============
# 转化到HSV色彩空间，以便识别红色区域
hsv = cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
# 红色空间1
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)
# 红色区间2
lower_red = np.array([170, 120, 70])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)
# 掩膜mask1，红色整体区间 = 红色区间1 + 红色区间2
mask1 = mask1 + mask2
cv2.imshow('mask1', mask1)
# 掩膜mask2，对mask1按位取反，获取mask1的反色图像
mask2 = cv2.bitwise_not(mask1)
# ===========图像C：背景中与前景红斗篷区域对应的位置图像============
C = cv2.bitwise_and(A, A, mask=mask1)
cv2.imshow('C', C)
# ===========图像D：抠除红斗篷区域的前景============
# 提取图像B中掩膜mask2指定的区域
D = cv2.bitwise_and(B, B, mask=mask2)
cv2.imshow('D', D)
# ===========图像E：图像C + 图像D============
E = C + D
cv2.imshow('E', E)
cv2.waitKey()
cv2.destroyAllWindows()
