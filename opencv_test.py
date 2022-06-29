import cv2
import numpy as np

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


