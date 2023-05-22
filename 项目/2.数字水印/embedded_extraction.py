import cv2
import numpy as np

lena = cv2.imread('../img/lena.bmp', 0)  # 读取图像
r, c = lena.shape   # 获取图像的宽和高
# =========嵌入过程=========
# Step 1：生成内部值都是254的数组
t1 = np.ones((r, c), dtype=np.uint8) * 254
# Step 2：获取原始载体图像的高7位，最低有效位清0
lsb0 = cv2.bitwise_and(lena, t1)
# Step 3：水印信息处理
w = cv2.imread('../img/watermark.bmp', 0)
# 将水印图像内的数值255处理为1，以便嵌入
wt = w.copy()
wt[w > 0] = 1
# Step 4：将水印图像wt嵌入lsb0内
wo = cv2.bitwise_or(lsb0, wt)
# ==========提取过程==========
# Step 5：生成内部都为1的数组
t2 = np.ones((r, c), dtype=np.uint8)
# Step 6：从载体内提取水印图像
ewb = cv2.bitwise_and(wo, t2)
# Step 7：将水印图像内的数值1处理为255，以便显示
ew = ewb
ew[ewb > 0] = 255
# ==========显示=============
cv2.imshow('lena', lena)    # 原始图像
cv2.imshow('watermark', w)  # 原始水印图像
cv2.imshow('wo', wo)        # 含水印载体
cv2.imshow('ew', ew)        # 提取的水印图像
cv2.waitKey()
cv2.destroyAllWindows()