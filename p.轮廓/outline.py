import cv2

im = cv2.imread("../img/1.png")
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,127,255,0)
# 查找轮廓
contours, hierarchy, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print(hierarchy)  # hierarchy代表的是轮廓的层析结构
print(contours)  # contours代表轮廓
# 绘制轮廓
img = cv2.drawContours(imgray, contours, -1, (0, 60, 255), 10)
# img = cv2.drawContours(thresh, contours,3,(0,255,0),6)
cv2.imshow("image", img)
cv2.waitKey(0)