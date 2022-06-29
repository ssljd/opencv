import cv2
import numpy as np

# 创建黑色背景
img = np.zeros((512,512,3), np.uint8)
# 绘制矩形
cv2.rectangle(img,(50,50),(180,180),(0,255,0),3)
cv2.imshow('FRAME',img)
cv2.waitKey(0)