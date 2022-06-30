# 绘图函数
- draw_line.py [code](draw_line.py)
- draw_rectangle.py [code](draw_rectangle.py)
- draw_circle.py [code](draw_circle.py)
- draw_ellipse.py [code](draw_ellipse.py)
- draw_word.py [code](draw_word.py)
## 画线
```
cv2.line(img, pt1, pt2, color, thickness, lineType, shift)
```
1）img：绘制图形的图像（背景图）  
2）pt1：直线起点坐标  
3）pt2：直线终点坐标  
4）color：形状的颜色。以RGB为例，需要传入一个元组，例如：（255,0,0）  
代表蓝色。对于灰度图只需要传入灰度值。  
5）thickness：画笔的粗细，线宽。  
6）lineType：线条的类型  
7）shift：点坐标的小数位数  
## 画矩形
```
cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)
```
1）img：绘制图形的图像（背景图）  
2）pt1：矩形框左上角坐标  
3）pt2：矩形框右下角坐标  
4）color：形状的颜色。以RGB为例，需要传入一个元组，例如：（255,0,0）  
代表蓝色。对于灰度图只需要传入灰度值。  
5）thickness：矩形边框的厚度，如果是负值，则填充整个矩形。  
6）lineType：边界线条的类型  
7）shift：点坐标的小数位数  
## 画圆
```
cv2.circle(img, pt1, pt2, color, thickness, lineType, shift)
```
1）img：绘制图形的图像（背景图）  
2）center：中心坐标  
3）radius：半径  
4）color：形状的颜色。以RGB为例，需要传入一个元组，例如：（255,0,0）  
代表蓝色。对于灰度图只需要传入灰度值。  
5）thickness：圆边框的厚度，如果是负值，则填充整个圆。  
6）lineType：边界线条的类型  
7）shift：点坐标的小数位数  
## 画椭圆
```
cv2.ellipse(img, pt1, pt2, color, thickness, lineType, shift)
```
1）img：绘制图形的图像（背景图）  
2）center：中心坐标  
3）axes：两个变量的元组，椭圆的长轴和短轴  
4）angle：椭圆旋转角度  
5）startAngle：椭圆弧的起始角度  
6）endAngle：椭圆弧的终止角度  
7）color：形状的颜色。以RGB为例，需要传入一个元组，例如：（255,0,0）  
代表蓝色。对于灰度图只需要传入灰度值。  
8）thickness：圆边框的厚度，如果是负值，则填充整个圆。  
9）lineType：边界线条的类型  
10）shift：点坐标的小数位数  
## 添加文字
```
cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
```
1）img：绘制图形的图像（背景图）  
2）text：绘制的文本字符串  
3）org：图像中文本字符串左下角的坐标。坐标表示两个值的元组  
4）font：字体类型  
5）fontScale：字体比例因子  
6）color：形状的颜色。以RGB为例，需要传入一个元组，例如：（255,0,0）  
代表蓝色。对于灰度图只需要传入灰度值。  
7）thickness：字体的粗细  
8）lineType：行的类型  
9）bottomLeftOrigin：如果为true，则图像数据原点位于左下角。否则，它位于左上角。  
