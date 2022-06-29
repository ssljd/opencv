# 形态学转换
## 腐蚀
```
cv2.erode(src, kernel, iteration)
```
1）src：输入图片  
2）kernel：方框的大小  
3）iteration：迭代的次数  
## 膨胀
```
cv2.dilate(src, kernel, iteration)
```
1）src：输入图片  
2）kernel：方框的大小  
3）iteration：迭代的次数 
## 运算
```
cv2.morphologyEx(src, op, kernel)
```
1）src：输入图片   
2）op：变化的方式
- cv2.MORPH_OPEN（开运算）
- cv2.MORPH_CLOSE（闭运算）
- cv2.MORPH_GRADIENT（形态学梯度）
- cv2.MORPH_TOPHAT（礼帽）
- cv2.MORPH_BALCKHAT（黑帽）
3）kernel：方框的大小  


