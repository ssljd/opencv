import cv2
import numpy as np

# 创建黑色背景
img = np.zeros((512,512,3), np.uint8)
# 绘制圆
cv2.circle(img,(250,250),100,(0,255,255),-1)
cv2.imshow('FRAME',img)
cv2.waitKey(0)