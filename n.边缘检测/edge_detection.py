#Canny边缘检测
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../img/IMG_20201017_101216f.jpg',0)

def nothing(x):#函数定义
    pass
cv2.namedWindow('edges')

cv2.createTrackbar('minVal','edges',0,255,nothing)
cv2.createTrackbar('maxVal','edges',0,255,nothing)
switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch,'edges',0,1,nothing)
while 1:
    min1 = cv2.getTrackbarPos("minVal", "edges")
    max1 = cv2.getTrackbarPos("maxVal", "edges")
    edges = cv2.Canny(img, min1, max1)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.imshow('edges',edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 'q':
        break
    # s = cv2.getTrackbarPos(switch, 'edges')
    # if s == 0:
    #     img[:] = 0
    # else:
    #     edges[:] = [min1,max1]