# 调色板
```
cv2.createTrackbar(trackbarName, windowName, value, count, onChange)
```
1）trackbarname：跟踪栏名称，创建的轨迹栏的名称  
2）windomNmme：窗口的名字  
3）value：指向整数变量的可选指针，该变量的值反映滑块的初始位置  
4）count：表示滑块可以达到的最大位置的值，最小位置始终为0  
5）onChange：指向每次滑块更改位置时要调用的函数的指针，有默认值0  
```
cv2.getTrackbarPos(trackbarName, windowName)
```
1）trackbarname：跟踪栏名称，创建的轨迹栏的名称  
2）windomNmme：窗口的名字  