import cv2

img = cv2.imread('../img/test.png',0)
ret,th = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(th,1,2)
cnt = contours[0]#cnt(轮廓)
M = cv2.moments(cnt)
# print(M)
#计算重心
Cx = float(M['m10']/M['m00'])
Cy = float(M['m01']/M['m00'])
print("轮廓重心的x轴点坐标：{}".format(Cx))
print("轮廓重心的y轴点坐标：{}".format(Cy))
#计算轮廓面积
area = cv2.contourArea(cnt)
print("轮廓的面积：{}".format(area))
#计算轮廓周长
perimeter = cv2.arcLength(cnt,True)
print("轮廓的周长：{}".format(perimeter))
#轮廓近似
epsilon = 0.01 * cv2.arcLength(cnt,True)#epsilon:原始轮廓到近似轮廓的最大距离
approx = cv2.approxPolyDP(cnt,epsilon,True)
img1 = cv2.imread("test.png")
cv2.polylines(img1, [approx], True, (0, 0, 255), 2)
cv2.imwrite('approxcurve3.jpg',img1)
#凸包
hull = cv2.convexHull(cnt,False,True,False)
print(hull)
#凸性检测
k = cv2.isContourConvex(cnt)
print("判断曲线是否为凸：{}".format(k))
#边界矩阵
x,y,w,h = cv2.boundingRect(cnt)#（x，y）为矩形左上角的坐标,（w，h）是矩形的宽和高
print("直边界矩形左上角坐标({},{}),宽：{},高：{}".format(x,y,w,h))
#无意义
img2 = cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow("img2",img2)
cv2.waitKey(0)
#轮廓的性质
aspect_ratio = float(w)/h
print("轮廓的宽高比：{}".format(aspect_ratio))