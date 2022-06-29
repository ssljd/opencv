import cv2
import numpy as np

img = cv2.imread("../img/1.png")
lower_reso = cv2.pyrDown(img)
higher_reso = cv2.pyrUp(img)
cv2.imshow("lower_reso",lower_reso)
cv2.imshow("higher_reso",higher_reso)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#使用金字塔进行图像融合
#出现错误
'''
# import numpy as sys
A = cv2.imread("../img/1.png")
B = cv2.imread("../img/2.png")
#对A生成高斯金字塔
G = A.copy()
gpA = [G]
for i in range(G):
    print(i)
    G = cv2.pyrDown(G)
    gpA.append(G)
cv2.imshow("gpA",gpA)
#对B生成高斯金字塔
G = B.copy()
gpB = [G]
for i in range(G):
    G = cv2.pyrDown(G)
    gpB.append(G)
#对A生成拉普拉斯金字塔
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)
#对B生成拉普拉斯金字塔
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)
#现在在每一层中添加图像的左右两半
#numpy.hstack（图普）
#取一系列数组并水平堆叠
#制作单个数组。
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2],lb[:,cols/2:]))
    LS.append(ls)
#现在重建
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_,LS[i])
#直接连接每一半的图像
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
cv2.imwrite("Pyramid_blending2.jpg",ls_)
cv2.imwrite("Direct_blending.jpg",real)