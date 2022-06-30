# 图像 
- img.py [code](img.py)
## 读入图像
```
cv2.imread(filename, flags)读入图像
```
>- 第一个参数filename：图片文件名（图片在此程序的工作路径）</p>

>- 第二个参数flags：告诉函数以何种方式读取图片
>>- cv2.IMREAD_COLOR或默认值：读入彩色图片，忽略alpha通道
>>- cv2.IMREAD_GRAYSCALE或0：读入灰色图片
>>- cv2.IMREAD_UNCHANGED：读入彩色图片，包括alpha通道
>>> $\color{red}alpha通道$：又称A通道，是一个8位的灰色通道，该通道用256级灰度来记录图像中的透明度信息，定义透明、不透明和半透明区域，其中黑表示全透明，白表示不透明，灰表示半透明

## 显示图像

```
cv2.imshow(winname, mat)显示图像
```
- 第一个参数winname：窗口的名称
- 第二个参数mat：图像
```
cv2.waitKey()键盘绑定函数
```
- 它的时间尺度是毫秒级
- 函数等待特定的几毫秒，看是否有键盘输入，在特定的几毫秒内，如果按下任意键，这个函数会返回按键的ASCII码值；如果没有键盘输入，返回值为-1
- 函数的参数设置为0，它将无限等待键盘输入
```
cv2.destroyAllWindows()
```
- 删除任何自己建立的窗口
```
cv2.destroyWindows()
```
- 删除特定窗口（括号内参数为指定删除的窗口名）
```
cv2.namedWindows(winname, flags)
```
>- 第一个参数winname：窗口的名称

>- 第二个参数flags：决定窗口是否可以调整大小
>>- cv2.WINDOW_AUTOSIZE：初始设定函数标签
>>- cv2.WINDOW_NORMAL：可以调整窗口大小
## 保存图像
```
cv2.imshow(filename, mat)保存图像
```
- 第一个参数filename：图片文件名（图片在此程序的工作路径）
- 第二个参数mat：图像