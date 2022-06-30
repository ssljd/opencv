# 图像平滑
- filter.py [code](filter.py)
- vague.py [code](vague.py)
## 2D卷积
```
cv2.filter2D(src, ddepth, kernel)
```
1）src：过滤器的源图像  
2）ddepth：输出图像的深度  
3）kernel：图像与之卷积的二维矩阵
## 图像模糊
### 平均
```
cv2.blur(src, ksize)
```
1）src：原图像  
2）ksize：核大小
### 方框滤波
```
cv2.boxFilter(src, ddepth, ksize, anchor, normalize, borderType)
```
1）src：原始图像  
2）ddepth：处理结果图像的图像深度  
3）ksize：滤波核的大小  
4）anchor：锚点  
5）normalize：滤波时是否进行归一化  
### 高斯模糊
```
cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)
```
1）src：输入图像  
2）ksize：高斯核大小  
3）sigmaX：X方向上的高斯核标准偏差  
4）sigmaY：Y方向上的高斯核标准差
### 中值模糊
```
cv2.medianBlur(src, ksize)
```
1）src：原图像  
2）ksize：滤波核大小
### 双边滤波
```
cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType)
```
1）src：输入图像  
2）d：滤波时选取的空间距离参数，表示以当前像素点为中心点的直径
3）sigmaColor：滤波处理时选取的颜色差值范围
4）sigmaSpace：坐标空间中的sigma值
5）borderType：边界样式