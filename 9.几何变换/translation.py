import cv2
import numpy as np

img = cv2.imread("../img/1.png", 0)
rows, cols = img.shape
img_move = cv2.warpAffine(img, (100, 50), (rows, cols))
cv2.imshow("img",img)
cv2.imshow("img_move",img_move)
cv2.waitKey(0)
cv2.destroyAllWindows()