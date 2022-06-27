import cv2
import numpy as np

img = cv2.imread("../img/test.png",0)
kernel = np.ones((5,5),np.uint8)  # 使用(5,5)的卷积核
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)  # 形态学梯度
while 1:
    cv2.imshow("image", img)
    cv2.imshow("gradient",gradient)
    if cv2.waitKey(1)&0xFF == 27:
        break
cv2.destroyAllWindows()