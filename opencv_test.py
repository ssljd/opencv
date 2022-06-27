import cv2
import numpy as np


#扩展缩放图像
'''
img = cv2.imread("1.png")
res = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
# height,width = img.shape[:2]
# res = cv2.resize(img,(2 * width,2 * height),interpolation=cv2.INTER_CUBIC)
while(1):
    cv2.imshow("image",img)
    cv2.imshow("res",res)
    if cv2.waitKey(1) & 0xFF == 27:#如果按键是esc,则关闭窗口
        break
cv2.destroyAllWindows()
'''
#平移图像
#下面代码有错误
'''
img = cv2.imread("1.png")
rows,cols = img.shape
img_move = cv2.warpAffine(img,(100,50),(rows,cols))
cv2.imshow("img",img)
cv2.imshow("img_move",img_move)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#旋转图像
'''
img = cv2.imread('1.png',0)
rows,cols = img.shape
#这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
#可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,0.6)

#第三个参数是输出图像的尺寸中心
dst = cv2.warpAffine(img,M,(cols,rows))
while(1):
    cv2.imshow('img',img)
    cv2.imshow('dst',dst)
    if cv2.waitKey(1)&0xFF == 27:
        break
cv2.destroyAllWindows()
'''
#仿射变换
'''
from matplotlib import pyplot as plt
img = cv2.imread('IMG_20201016_212915.jpg')
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))#dst是img经过仿射变换后得到的图像
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
'''
#透视变换
'''
from matplotlib import pyplot as plt
img = cv2.imread("IMG_20201027_162850.jpg")
print(img.shape)
rows,cols,ch = img.shape
print(rows)
print(cols)
print(ch)
pts1 = np.float32([[456,365],[3280,365],[368,4126],[3890,4278]])
pts2 = np.float32([[0,0],[2000,0],[0,2000],[2000,2000]])
M = cv2.getPerspectiveTransform(pts1,pts2)#dst是img经过透视变换后得到的图像
dst = cv2.warpPerspective(img,M,(2000,2000))
plt.subplot(131),plt.imshow(img),plt.title('Input')
plt.subplot(133),plt.imshow(dst),plt.title('Output')
plt.show()
'''

#OpenCV检测程序效率
'''
img = cv2.imread("1.png")
e1 = cv2.getTickCount()
for i in range(1,49,2):
    img = cv2.medianBlur(img,i)
    print(i)
    cv2.imshow("image",img)
    cv2.waitKey(1000)
e2 = cv2.getTickCount()
t = (e2 - e1) / cv2.getTickFrequency()
print(t)
'''
#*****************OpenCV图像处理*******************#
#物体跟踪
'''
cap = cv2.VideoCapture(0)
cap.isOpened()
# fourcc = cv2.cv.FOURCC(*'XVID')
while(1):
    #获取每一帧
    ret,frame = cap.read()
    #转换到HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #设定蓝色的阈值
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([130,255,255])
    #根据阈值构建掩模
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    #对原图像和掩模进行位运算
    res = cv2.bitwise_and(frame,frame,mask= mask)
    #显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5)&0xFF
    if k == 27:
        break
#关闭窗口
cap.release()
cv2.destroyAllWindows()
'''
#HSV值获取
'''
green = np.uint8([[[0,255,0]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print("green's hsv",hsv_green)
#下面代码还有错误
red = np.uint([[[0,0,255]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
print("red's hsv",hsv_red)
blue = np.uint([[[255,0,0]]])
hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
print("blue's hsv",hsv_blue)
'''
#图像阈值
#简单阈值
'''
from matplotlib import pyplot as plt
img = cv2.imread('IMG_20201104_180926.jpg',0)
ret,thresh1 = cv2.threshold(img,126,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,126,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,126,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,126,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,126,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
'''
#自适应阈值
'''
from matplotlib import pyplot as plt
img = cv2.imread('IMG_20201016_212915.jpg',0)
#中值滤波
img = cv2.medianBlur(img,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
titles = ['Original Image','Global Thresholding (v = 127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']
images = [img,th1,th2,th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
'''
cv2.copyMakeBorder()
#Otsu's 二值化
'''
from matplotlib import pyplot as plt
img = cv2.imread('IMG_20201027_162850.jpg',0)

#全局阈值
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#Otsu's 阈值
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#在高斯核除去噪音之后Otsu二值化
#(5,5)为高斯核的大小，0为标准差
blur = cv2.GaussianBlur(img,(5,5),0)
#阈值一定要设为0
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#绘制所有图像及其直方图
images = [img,0,th1,
          img,0,th2,
          blur,0,th3]
print(th1)
print(th2)
print(th3)
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# 这里使用了 pyplot 中画直方图的方法，plt.hist, 要注意的是它的参数是一维数组
# 所以这里使用了（numpy）ravel 方法，将多维数组转换成一维，也可以使用 flatten 方法
#ndarray.flat 1-D iterator over an array.
#ndarray.flatten 1-D array copy of the elements of an array in row-major order.
for i in range(3):
    plt.subplot(3,3,i * 3 + 1),plt.imshow(images[i * 3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i * 3 + 2),plt.hist(images[i * 3].ravel(),256)
    plt.title(titles[i * 3 + 1]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i * 3 + 3),plt.imshow(images[i * 3 + 2],'gray')
    plt.title(titles[i * 3 + 2]),plt.xticks([]),plt.yticks([])
plt.show()
'''
#图像模糊(图像平滑）：卷积
'''
from matplotlib import pyplot as plt
img = cv2.imread('1.png')
kernel = np.ones((5,5),np.float32) / 25
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]),plt.yticks([])
plt.show()
'''
#图像模糊(图像平滑）：平均,高斯模糊，中值模糊，双边滤波
'''
from matplotlib import pyplot as plt
img = cv2.imread('1.png')
blur = cv2.blur(img,(5,5))#平均
blur1 = cv2.GaussianBlur(img,(5,5),0)#高斯模糊
median = cv2.medianBlur(img,5)#中值模糊
blur2 = cv2.bilateralFilter(img,9,75,75)#双边滤波
# dst = cv2.filter2D(img,-1,blur)
plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(232),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]),plt.yticks([])
plt.subplot(233),plt.imshow(blur1),plt.title('Gaussian')
plt.xticks([]),plt.yticks([])
plt.subplot(234),plt.imshow(img),plt.title('Median')
plt.xticks([]),plt.yticks([])
plt.subplot(235),plt.imshow(img),plt.title('Bilateral')
plt.xticks([]),plt.yticks([])
plt.show()
'''
#形态学转换
'''
img = cv2.imread("test.png",0)
kernel = np.ones((5,5),np.uint8)#使用(5,5)的卷积核
erosion = cv2.erode(img,kernel,iterations=1)#腐蚀
dilation = cv2.dilate(erosion,kernel,iterations=1)#膨胀
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)#开运算
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)#闭运算
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)#形态学梯度
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)#礼帽
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)#黑帽
# black = cv2.morphologyEx(tophat,cv2.MORPH_OPEN,kernel)
while(1):
    cv2.imshow("image", img)
    cv2.imshow("erosion", erosion)
    cv2.imshow("dilation", dilation)
    cv2.imshow("opening", opening)
    cv2.imshow("closing", closing)
    cv2.imshow("gradient",gradient)
    cv2.imshow("tophat",tophat)
    cv2.imshow("blackhat",blackhat)
    # cv2.imshow("black",black)
    if cv2.waitKey(1)&0xFF == 27:
        break
cv2.destroyAllWindows()
'''
#图像梯度
'''
from matplotlib import pyplot as plt
img = cv2.imread("test.png",0)

#cv2.CV_64F 输出图像的深度（数据类型），可以使用-1, 与原图像保持一致 np.uint8
laplacian = cv2.Laplacian(img,cv2.CV_64F)#拉普拉斯算子
#参数1，0为只在x方向求一阶导数，最大可以求二阶导数
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#参数0，1为只在y方向求一阶导数，最大可以求二阶导数
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('Original'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')
plt.title('Soble X'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')
plt.title('Soble Y'),plt.xticks([]),plt.yticks([])
plt.show()

'''
#Canny边缘检测
#已理解边缘检测的含义，但用滑度调调节阈值还未解决
'''
from matplotlib import pyplot as plt

img = cv2.imread('IMG_20210509_022632.jpg',0)

def nothing(x):#函数定义
    pass
cv2.namedWindow('edges')

cv2.createTrackbar('minVal','edges',0,255,nothing)
cv2.createTrackbar('maxVal','edges',0,255,nothing)
switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch,'edges',0,1,nothing)
while(1):
    min1 = cv2.getTrackbarPos("minVal", "edges")
    max1 = cv2.getTrackbarPos("maxVal", "edges")
    edges = cv2.Canny(img, min1, max1)
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    cv2.imshow('edges',edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 'q':
        break
    s = cv2.getTrackbarPos(switch, 'edges')
    if s == 0:
        img[:] = 0
    else:
        edges[:] = [min1,max1]
'''
#图像金字塔
'''
#原理
img = cv2.imread("1.png")
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
A = cv2.imread("1.png")
B = cv2.imread("2.png")
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
'''
#*****************OpenCV中的轮廓*****************#
#初始轮廓
'''
im = cv2.imread("1.png")
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
#查找轮廓
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print(hierarchy)#hierarchy代表的是轮廓的层析结构
print(contours)#contours代表轮廓
#绘制轮廓
img = cv2.drawContours(imgray,contours,-1,(0,60,255),10)
# img = cv2.drawContours(thresh,contours,3,(0,255,0),6)
cv2.imshow("image",img)
cv2.waitKey(0)
'''
#轮廓特征
'''
img = cv2.imread('test.png',0)
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
'''
#*****************直方图*************************#
#绘制直方图
'''
from matplotlib import pyplot as plt
img = cv2.imread('IMG_20201030_184553.jpg')
plt.hist(img.ravel(),256,[0,256])
color = ('b','g','r')
# 对一个列表或数组既要遍历索引又要遍历元素时
# 使用内置 enumerrate 函数会有更加直接，优美的做法
#enumerate 会将数组或列表组成一个索引序列。
# 使我们再获取索引和索引内容的时候更加方便
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color=col)
#     plt.xlim([0,256])
#创建掩膜
mask = np.zeros(img.shape[:2],np.uint8)
mask[0:4000,0:4600] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)
#计算带有和不带有掩膜的直方图
#检查掩膜的第三个参数
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.show()
'''
#直方图均衡化
'''
from matplotlib import pyplot as plt
img = cv2.imread("IMG_20201224_162831.jpg",0)
#flatten()将数组变成一堆
hist,bins = np.histogram(img.flatten(),256,[0,256])
#计算累积分布图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
#构建Numpy掩膜数组，cdf为数组，当数组元素为0时，掩盖
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
#对被掩盖的元素赋值，这里赋值为0
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img1 = cdf[img]
plt.plot(cdf_normalized,color='b')
plt.hist(img1.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc='upper left')
plt.show()
'''
#OpenCV中的直方图均衡化
'''
img = cv2.imread('IMG_20210509_022632.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
#创建一个CLAHE对象
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(img)
while(1):
    cv2.imshow("cl1", cl1)
    cv2.imshow("res", res)#并排堆叠图像
    cv2.imwrite('clahe.jpg',cl1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
'''
#绘制2D直方图
'''
from matplotlib import pyplot as plt
img = cv2.imread("IMG_20201030_184553.jpg")
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
plt.imshow(hist,interpolation='nearest')
plt.show()
from time import clock
import sys
import video
if __name__ == '__main__':
    #构建HSV颜色地图
    hsv_map = np.zeros((180,256,3),np.uint8)
    # np.indices 可以返回由数组索引构建的新数组。
    # 例如：np.indices（（3,2））；其中（3,2）为原来数组的维度：行和列。
    # 返回值首先看输入的参数有几维：（3,2）有 2 维，所以从输出的结果应该是
    # [[a],[b]], 其中包含两个 3 行，2 列数组。
    # 第二看每一维的大小，第一维为 3, 所以 a 中的值就 0 到 2（最大索引数），
    # a 中的每一个值就是它的行索引；同样的方法得到 b（列索引）
    # 结果就是
    # array([[[0, 0],
    # [1, 1],
    # [2, 2]],
    #
    # [[0, 1],
    # [0, 1],
    # [0, 1]]])
    h,s = np.indices(hsv_map.shape[:2])
    hsv_map[:,:,0] = h
    hsv_map[:,:,1] = s
    hsv_map[:,:,2] = 255
    hsv_map = cv2.cvtColor(hsv_map,cv2.COLOR_HSV2BGR)
    cv2.imshow('hist',hsv_map)
    cv2.waitKey(0)
    cv2.namedWindow('hist',0)
    hist_scale = 10
    def set_scale(val):
        global hist_scale
        hist_scale = val
    cv2.createTrackbar('scale','hist',hist_scale,32,set_scale)
    try:fn = sys.argv[1]
    except:fn = 0
    cam = video.create_capture()
'''
#直方图反向投影
'''
from matplotlib import pyplot as plt
#roi是我们需要找到的对象或对象的区域
roi = cv2.imread('1.png')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#目标是我们搜索的图像
target = cv2.imread('2.png')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
#使用calcHist查找直方图。也可以用np.historogram2d完成
M = cv2.calcHist([hsv],[0,1],None,[200,256],[0,200,0,256])
I = cv2.calcHist([hsvt],[0,1],None,[200,256],[0,200,0,256])
R = M/I
# print(R)
h,s,v = cv2.split(hsvt)
B = R[h.ravel(),s.ravel()]
print(B)
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])
print(B)
#使用圆盘算子做卷积
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
B = cv2.filter2D(B,-1,disc)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
ret,thresh = cv2.threshold(B,50,255,0)
plt.subplot(111),plt.imshow(thresh,'gray')
plt.title("B")
plt.show()
'''
#****************图像变换*************************#
#傅里叶变换
#Numpy中的傅里叶变换
'''
from matplotlib import pyplot as plt
img = cv2.imread("C:/Users/sljd/Desktop/Code_library/python/image/IMG_20201224_162831.jpg",0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# # print(fshift)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
rows,cols = img.shape
crow,ccol = rows/2 , cols/2
# print(type(crow))
fshift[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 0
f_ishift = np.fft.ifftshift(fshift)#逆平移操作
img_back = np.fft.ifft2(f_ishift)#FFT逆变换
img_back = np.abs(img_back)#取绝对值
plt.subplot(221),plt.imshow(img,'gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum,'gray')
plt.title('Magnitude spectrum'),plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.imshow(img_back,'gray')
plt.title('Image after HPF'),plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(img_back)
plt.title('Result in JET'),plt.xticks([]),plt.yticks([])
plt.show()
'''
#OpenCV中的傅里叶变换
'''
from matplotlib import pyplot as plt
img = cv2.imread("C:/Users/sljd/Desktop/Code_library/python/image/IMG_20201224_162831.jpg",0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
rows,cols = img.shape
crow,ccol = rows/2,cols/2
#首先创建一个遮罩，中心正方形为1，其余为0
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow-30):int(crow+30),int(ccol-30):int(ccol+30)] = 1
#应用掩模和逆DFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0],img_back[:, :, 1])
plt.subplot(131),plt.imshow(img,'gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum,'gray')
plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(img_back,'gray')
plt.title('Image after HPF'),plt.xticks([])
plt.show()
'''
#DFT性能优化
'''
img = cv2.imread('1.png',0)
rows,cols = img.shape
print(rows,cols)
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
print(nrows,ncols)
nimg = np.zeros((nrows,ncols))
nimg[:rows,:cols] = img
print(nimg.shape)
'''
#拉普拉斯算子（高通滤波器）
'''
from matplotlib import pyplot as plt
#无标度参数的简单平均滤波器
mean_filter = np.ones((3,3))
#创建一个高斯滤波器
x = cv2.getGaussianKernel(5,10)
#x,T为矩阵转置
gaussian = x*x.T
#不同的边缘检测滤波器
#x方向的Schar
scharr = np.array([[-3,0,3],
                   [-10,0,10],
                   [-3,0,3]])
#x方向sobel
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,-1]])
#y方向sobel
sobel_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])
#拉普拉斯
laplacian = np.array([[0,1,0],
                      [1,-4,1],
                      [0,1,0]])
filters = [mean_filter,gaussian,laplacian,sobel_x,sobel_y,scharr]
filter_name = ['mean_filter','gaussian','laplacian','sobel_x','sobel_y','scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],'gray')
    plt.title(filter_name[i]),plt.xticks([]),plt.yticks([])
plt.show()
'''
#****************模板匹配**************************#
#OpenCV中的模板匹配
'''
from matplotlib import pyplot as plt
img = cv2.imread('IMG_20201017_101216.jpg',0)
img2 = img.copy()
template = cv2.imread('IMG_20201017_101216f.jpg',0)
w,h = template.shape[::-1]
print(w)
print(h)
#所有6种比较方法都列在一个列表中
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
#exec 语句用来执行储存在字符串或文件中的 Python 语句。
# 例如，我们可以在运行时生成一个包含 Python 代码的字符串，然后使用 exec 语句执行这些语句。
#eval 语句用来计算存储在字符串中的有效 Python 表达式
    method = eval(meth)
    # 应用模板匹配
    res = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    if methods in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w,top_left[1] + h)
    cv2.rectangle(img,top_left,bottom_right,255,2)
    plt.subplot(121),plt.imshow(res,'gray')
    plt.title('Matching Result'),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img,'gray')
    plt.title('Detected Point'),plt.xticks([]),plt.yticks([])
    plt.suptitle(meth)
    plt.show()
'''
#多对象的模板匹配
'''
from matplotlib import pyplot as plt
img_rgb = cv2.imread('1.png')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template = cv2.imread('1f.png',0)
w,h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8

loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
cv2.imshow('res',img_rgb)
cv2.waitKey(0)
'''
#****************Hough直线变换*********************#
#OpenCV中的霍夫变换
'''
img = cv2.imread('IMG_20210720_194814.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize=3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("img",img)
cv2.waitKey(0)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    while(1):
        cv2.imshow("img",img)
        if cv2.waitKey(1)&0xFF == 27:
            break
'''
#Hough圆环变换
'''
img = cv2.imread('3.png',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=60,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
print(circles)
for i in circles[0,:]:
    #画外圆
    print(i)
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #画圆的中心
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.HoughCircles（图像，方法，dp，minDist，圆，param1，param2，minRadius，maxRadius）
#参数：
#图像–8位，单通道，灰度输入图像。
# 返回结果为 找到的圆的输出向量。每个向量被编码为
#三元素浮点向量（x，y，半径）。
#找到的圆的输出序列。
#方法–要使用的检测方法。目前，唯一实现的方法是
#dp–累加器分辨率与图像分辨率的反比。
#例如，如果dp=1，累加器的分辨率与输入图像相同。
#如果dp=2，则蓄能器的宽度和高度为原来的一半。
#minDist–检测到的圆的中心之间的最小距离。
#如果参数太小，可能会错误地生成多个相邻圆
#除了一个真的外还发现了。如果太大，可能会漏掉一些圆。
#它是传递给Canny（）边缘检测器的两个阈值中的较高阈值
#（下一个小两倍）。
#它是检测阶段圆心的累加器阈值。
#它越小，检测到的假圆就越多。圈子，
#对应于较大的累加器值，将首先返回。
#minRadius–最小圆半径。
#maxRadius–最大圆半径。
'''
#*********使用GrabCut算法进行交互式前景提取************#
'''
from matplotlib import pyplot as plt
img = cv2.imread('IMG_20210509_022632.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (123,37,320,580)
#函数的返回值是更新的mask,bgdModel,fgdModel
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2)|(mask == 0),0,1).astype('uint8')
img = img * mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
#newmask是我手动标记的掩码图像
newmask = cv2.imread('Inkednewmask_LI.png',0)
#无论在何处标记为白色（确定前景），更改mask=1
#无论在何处标记为黑色（确定背景），更改mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask,bgdModel,fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
mask = np.where((mask == 2)|(mask == 0),0,1).astype('uint8')
img = img * mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
'''
#***************图像特征提取与描述********************#
#Harris角点检测
'''
img = cv2.imread('test.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为GRAY格式
gray = np.float32(gray)
#输入图像必须是float32，最后一个参数在0.04到0.05之间
dst = cv2.cornerHarris(gray,2,3,0.04)
#结果是扩大标记的角落，不重要
# dst = cv2.dilate(dst,None)
#阈值为最佳值时，它可能会因图像而异
img[dst>0.01*dst.max()] = [0,80,200]#控制角点的颜色
cv2.imshow("dst",img)
if cv2.waitKey(0)&0xFF == 27:
    cv2.destroyAllWindows()
'''
#亚像素级精确度的角点
'''
img = cv2.imread('test.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##找到Harris角
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)#扩大标记的角落
ret,dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
#查找质心
#connectedComponentsWithStats（InputArray映像、OutputArray标签、OutputArray统计信息、，
#OutputArray质心，int connectivity=8，int ltype=）
ret,labels,stats,centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.001)
#cv2.cornerSubPix（图像、角点、winSize、zeroZone、条件）在搜索区域中部死亡区域的一半大小
#下面的公式中的求和没有完成。有时也会用到
#避免自相关矩阵可能出现的奇异性。值（-1，-1）
#表示没有这样的大小。
# 返回值由角点坐标组成的一个数组（而非图像）
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#现在画它们
res = np.hstack((centroids,corners))
#np.int0 可以用来省略小数点后面的数字（非四㮼五入）
res = np.int0(res)
img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]
cv2.imshow("subpixel5",img)
cv2.waitKey(0)
'''
#Shi_Tomasi角点检测&适合于跟踪的图像特征
'''
from matplotlib import pyplot as plt
img = cv2.imread('test1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,5,0.01,6)#适合目标跟踪中使用
#返回的结果是[[311., 250.]]两层括号的数组
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()
'''
#SIFT(Scale-Invariant Feature Transform)
'''
img = cv2.imread('IMG_20201017_101216f.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
# kp,des = sift.compute(gray,kp)
img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('images',img)
cv2.waitKey(0)
'''
#SURF(Speeded-Up Robust Features)
'''
from matplotlib import pyplot as plt
img = cv2.imread('test.png',0)
#创建SURF对象，可以在此处指定参数，也可以稍后指定
surf = cv2.xfeatures2d.SIFT_create()
#直接查找关键点和描述符
kp,des = surf.detectAndCompute(img,None)
print(len(kp))
print(surf.descriptorSize())#输出关键点描述符的大小
img1 = cv2.drawKeypoints(img,kp,None,(0,255,0),4)
plt.imshow(img1),plt.show()
'''
#角点检测的FAST算法
'''
img = cv2.imread('test.png',0)
#使用默认值初始化快速对象
fast = cv2.FastFeatureDetector_create()
#找到并画出关键点
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,img,color=(255,0,0))
cv2.imshow("img2",img2)
cv2.waitKey(0)
'''

#拍照
'''
cv2.VideoCapture.open()
X = 1
count = 1
def test():
    while True:
        # 调用摄像头
        cam = cv2.VideoCapture(2)
        while True:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            cv2.waitKey(1)
            # 加一个鼠标点击事件，frame传给了OnMouseAction的param
            # 鼠标左键点击相框，相当于按下快门
            cv2.setMouseCallback("test", OnMouseAction, frame)
            global X
            if X == 2:
                X = 1
                break
            if cv2.waitKey(1) & 0xFF == 27:  # 如果按下esc键，则跳出函数，停止拍照
                return
        cam.release()
        cv2.destroyAllWindows()
def OnMouseAction(event,x,y,flags,param):
    #cv2.EVENT_LBUTTONDOMN左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        global count
        cv2.imwrite(f"{count}.png",param) #保存图片
        count += 1
        global X
        X = 2
test()
'''

''''''
import numpy as np
import cv2
cap = cv2.VideoCapture('show.flv')
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 250,90,400,125 # simply hardcoded the values
track_window = (c, r, w, h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2', img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg", img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()


# import numpy as np
# import cv2
cap = cv2.VideoCapture('show.flv')
# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
r,h,c,w = 250,90,400,125 # simply hardcoded the values
track_window = (c, r, w, h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi], [0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()


