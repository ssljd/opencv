import cv2
import numpy as np

# 创建黑色背景
img = np.zeros((512,512,3), np.uint8)
# 绘制线
cv2.line(img, (50,50), (466,466), (255,255,0), 5)
cv2.imshow('FRAME',img)
cv2.waitKey(0)