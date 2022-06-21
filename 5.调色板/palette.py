import cv2
import numpy as np

def nothing(x):
    pass
#创建一副黑色图像
img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow("image")
cv2.createTrackbar('Red','image',0,255,nothing)
cv2.createTrackbar('Green','image',0,255,nothing)
cv2.createTrackbar('Blue','image',0,255,nothing)
switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1)&0xFF
    if k == 'q':
        break
    r = cv2.getTrackbarPos('Red','image')
    g = cv2.getTrackbarPos('Green','image')
    b = cv2.getTrackbarPos('Blue','image')
    s = cv2.getTrackbarPos(switch,'image')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
cv2.destroyWindow('image')




