# 图像阈值
- simple_threshold.py [code](simple_threshold.py)
- adaption_threshold.py [code](adaption_threshold.py)
- Otsu_threshold.py [code](Otsu_threshold.py)
## 简单阈值
```
cv2.threshold(src, thresh, maxval, type)
```
1）src：原图像  
2）thresh：分类的阈值  
3）maxval：高于（低于）阈值时赋予的新值（最大值）  
4）type：划分的时候使用的是什么类型的算法
- cv2.THRESH_BINARY（黑白二值）
> dst(x, y) = $\begin{cases} maxVal, &if\ src(x,y)\ >\ thresh\\ 0, &otherwise \end{cases}$
- cv2.THRESH_BINARY_INV（黑白二值反转）
> dst(x, y) = $\begin{cases} 0, &if\ src(x,y)\ >\ thresh\\ maxVal, &otherwise \end{cases}$
- cv2.THRESH_TRUNC （得到的图像为多像素值）
> dst(x, y) = $\begin{cases} threshold, &if\ src(x,y)\ >\ thresh\\ src(x,y), &otherwise \end{cases}$
- cv2.THRESH_TOZERO（超过阈值被置位0）
> dst(x, y) = $\begin{cases} src(x,y), &if\ src(x,y)\ >\ thresh\\ 0, &otherwise \end{cases}$
- cv2.THRESH_TOZERO_INV（低于阈值被置位0）
> dst(x, y) = $\begin{cases} 0, &if\ src(x,y)\ >\ thresh\\ src(x,y), &otherwise \end{cases}$
## 自适应阈值
```
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
```
1）src：灰度图像  
2）maxValue：满足条件的像素点需要设置的灰度值  
3）adaptiveThreshold：自适应阈值算法  
- cv2.ADAPTIVE_THRESH_MEAN_C（阈值取自相邻区域的平均值）
- cv2.ADAPTIVE_THRESH_GAUSSIAN_C（阈值取值相邻区域的加权和，权重为一个高斯窗口）  

4）thresholdType：二值化方法   
- cv2.THRESH_BINARY
- cv2.THRESH_BINARY_INV  

5）blockSize：要分成的区域大小，上面的N值，一般取奇数   
6）C：每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值
## Otsu二值化
> $\sigma$$_w^2$(t) = q$_1$(t)$\sigma$$_1^2$(t) + q$_2$(t)$\sigma$$_2^2$(t)
>> q$_1$(t) = $\sum_{i=1}^t$P(i)  &  q$_1$(t) = $\sum_{i=t+1}^I$P(i)</p>
>> u$_1$(t) = $\sum_{i=1}^t$$\frac{iP(i)}{q1(t)}$  &  u$_2$(t) = $\sum_{i=t+1}^I$$\frac{iP(i)}{q2(t)}$</p>
>> $\sigma$$_1^2$(t) = $\sum_{i=1}^t$[i - u$_1$(t)]$^2$$\frac{P(i)}{q1(t)}$  &  $\sigma$$_2^2$(t) = $\sum_{i=t+1}^t$[i - u$_1$(t)]$^2$$\frac{P(i)}{q2(t)}$
