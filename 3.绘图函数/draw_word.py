import cv2
import numpy as np

# 创建黑色背景
img = np.zeros((512,512,3), np.uint8)
# 添加文字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,300),font,4,(0,0,255),2)
cv2.imshow('FRAME',img)
cv2.waitKey(0)