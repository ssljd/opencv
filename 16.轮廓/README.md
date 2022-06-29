# 轮廓
## 什么是轮廓
轮廓可以简单认为成将连续的点（连着边界）连在一起的曲线，具有相同的颜色或者灰度。轮廓在形状分析和物体检测和识别中很有用。
- 为了更加准确，要使用二值化图像。在寻找轮廓之前，要进行阈值化处理或者 Canny 边界检测。
- 查找轮廓的函数会修改原始图像。如果你在找到轮廓之后还想使用原始图像的话，你应该将原始图像存储到其他变量中。
- 在 OpenCV 中，查找轮廓就像在黑色背景中超白色物体。你应该记住，要找的物体应该是白色而背景应该是黑色
## 绘制轮廓
```
cv2.drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset)
```
1）image：输入图像  
2）contours：轮廓本身  
3）contourIdx：指定绘制轮廓  
4）color：颜色  
5）thickness：轮廓线的宽度  
6）lineType：轮廓线的类型  
7）hierarchy：  
8）maxLevel：  
9）offset：
## 寻找轮廓
```
cv2.findContours(image, mode, method,contours, hierarchy, offset)
```
1）image：寻找轮廓的图像  
2）mode：轮廓的检索模式
- cv2.RETR_EXTERNAL：表示只检测外轮廓
- cv2.RETR_LIST：检测的轮廓不建立等级关系
- cv2.RETR_CCOMP：建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息
- cv2.RETR_TREE：建立一个等级树结构的轮廓

3）method：轮廓的近似方法  
- cv2.CHAIN_APPROX_NONE：存储所有的轮廓点，相邻的两个点的像素位置差不超过1
- cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素
- cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain近似算法

4）contours：轮廓本身  
5）hierarchy：轮廓对应的属性    
6）offset：
## 矩
```
cv2.moments(cnt)
```
- cnt=contours[0]，计算得到的矩以一个字典的形式返回</p>

计算对象的重心公式如下：
> C$_x$ = $\frac{M10}{M00}$, C$_y$ = $\frac{M01}{M00}$
## 轮廓面积
```
cv2.contourArea(cnt)
```
- cnt=contours[0]，计算得到轮廓的面积
## 轮廓周长
```
cv2.arcLength(cnt, True)
```
- cnt=contours[0]，计算得到轮廓的周长，第二参数可以用来指定对象的形状是闭合的（True）
## 轮廓近似
```
cv2.polylines(image, pts, isClosed, color, thickness)
```
1）image：绘制的图像  
2）pts：多边形曲行数组  
3）npts：多边形顶点计数器阵列  
4）ncontours：曲行数量  
5）isClosed：指示绘制的折线是否  
6）color：折线的颜色  
7）thickness：折线边的厚度  
## 凸包
```
cv2.convexHull(points, hull, clockwise, returnPoint)
```
1）points：传入的轮廓
2）hull：输出，通常不需要
3）clockwise：方向标志。如果设置为True，输出的凸包是顺时针方向的。否则为逆时针方向。
4）returnPoints：默认值为True。它会返回凸包上点的坐标。如果设置为False，就会返回于凸包点对应的轮廓上的点。