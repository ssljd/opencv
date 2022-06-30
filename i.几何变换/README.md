# 几何变换
- affine.py [code](affine.py)
- rotate.py [code](rotate.py)
- perspective.py [code](perspective.py)
- translation.py [code](translation.py)
- Extended_zoom.py [code](Extended_zoom.py)
### 扩展缩放
```
cv2.resize(src, dsize, fx, fy, interpolation)
```
1）src：输入图像  
2）dsize：代表期望的输出图像大小尺寸  
3）fx：代表水平方向上（图像宽度）的缩放系数  
4）fy：代表竖直方向上（图像高度）的缩放系数，另外，如果dsize被设置为0（None），则按fx与fy与原始图像大小相乘得到输出图像尺寸大小    
5）interpolation：插值方式，默认选择线性插值，越复杂的插值方式带来的改变和差异越大  
   - cv2.INTER_NEAREST：最临近插值算法
   - cv2.INTER_LINEAR：线性插值算法
   - cv2.INTER_CUBIC：双立方插值算法
   - cv2.INTER_AREA：区域插值算法
   - cv2.INTER_LANCZOS4：Lanczos插值（超过8x8邻域的插值算法）
   - cv2.INTER_MAX：用于插值的掩膜板
   - cv2.WARP_FILL_OUTLIERS：标志位，用于填充目标图像的像素值
   - cv2.WARP_INVERSE_MAP：标志位，反变换
### 平移
```
cv2.warpAffine(src, M, dsize, flags, borderMode, borderValue)
```
1）src：输入图像  
2）M：变换矩阵  
3）dsize：代表期望的输出图像大小尺寸   
4）flags：插值方法的组合  
5）borderMode：边界像素模式  
6）borderValue：边界填充值；默认情况下，为0  
### 旋转
```
cv2.getRotationMatrix2D(center, angle, scale)
```
1）center：旋转中心  
2）angle：旋转角度  
3）scale：旋转后的缩放比例  
### 仿射变换
```
cv2.getAffineTransform(pst1, pst2)
```
1）pst1：原图像三个点的坐标   
2）pst2：原图像三个点在变换后相应的坐标  
### 透视变换
```
cv2.getPerspectiveTransform(src, dst) -> retval
```
1）src：源图像中待测矩阵的四点坐标  
2）sdt：目标图像中矩阵的四点坐标  
返回由源图像中矩形到目标图像矩形变换的矩阵

