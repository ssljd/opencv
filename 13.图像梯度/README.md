# 图像梯度
## Sobel算子
```
cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta, borderType)
```
1）src：输入图像
2）ddepth：输出图像的深度
3）dx：x方向上的求导阶数
4）dy：y方向上的求导阶数
5）ksize：核的大小
6）scale：计算导数值时所采用的缩放因子
7）delta：加在目标图像dst上的值
8）borderType：边界样式
> $\Delta$src = $\partial$$^2$src/$\partial$x$^2$ + $\partial$$^2$src/$\partial$y$^2$
## Scharr算子