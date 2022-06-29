import cv2

# 以灰度模式读入图像
img1 = cv2.imread('../img/1.png', 0)
img2 = cv2.imread('../img/1.png', cv2.IMREAD_GRAYSCALE)
# 读入彩色图像
img3 = cv2.imread('../img/1.png')
img4 = cv2.imread('../img/1.png', cv2.IMREAD_COLOR)
img5 = cv2.imread('../img/1.png', cv2.IMREAD_UNCHANGED)
# 显示图像
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
cv2.imshow('img5', img5)
cv2.waitKey()   # 键盘绑定函数（具体解释看本目录的README）
cv2.destroyAllWindows()     # 删除任何建立的窗口

# 保存图像
cv2.imwrite('save.png', img1)