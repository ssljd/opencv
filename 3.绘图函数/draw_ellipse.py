import cv2
import numpy as np

# 创建黑色背景
img = np.zeros((512,512,3), np.uint8)
# 绘制半椭圆
cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
cv2.imshow('FRAME',img)
cv2.waitKey(0)