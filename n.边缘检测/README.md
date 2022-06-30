# 边缘检测
- edge_detection.py [code](edge_detection.py)
## 去除噪声
- 使用高斯滤波器去除噪声
## 计算图像梯度
- 对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数
> $$Edge_Gradient(G) = \sqrt{G_{x}^{2} + G_{y}^{2}}$$
> $$Angle(\theta) = \tan^{-1} \left(\frac{G_{x}}{G_{y}} \right)$$
## 非极大值抑制
- 在获得梯度的方向和大小之后，应该对整幅图像做一个扫描，去除那些非边界上的点。对每一个像素进行检查，看这个点的梯度是不是周围具有相同梯度方向的点中最大的
## 滞后阈值
- 设置两个阈值：minVal 和 maxVal。当图像的灰度梯度高于 maxVal 时被认为是真的边界，那些低于 minVal 的边界会被抛弃。如果介于两者之间的话，就要看这个点是否与某个被确定为真正的边界点相连，如果是就认为它也是边界点，如果不是就抛弃
## Canny边界检测
```
cv2.Canny(image, threshold1, threshold2, apertureSize, L2gradient)
```
1）image：输入图像  
2）threshold1：处理过程中的第一个阈值  
3）threshold2：处理过程中的第二个阈值  
4）apertureSize：Sobel算子的孔径大小  
5）L2gradient：计算图像梯度幅度的标识  