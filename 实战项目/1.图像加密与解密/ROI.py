import cv2
import numpy as np

lena = cv2.imread('../img/lena.bmp', 0)  # 读取图像
r, c = lena.shape   # 获取图像的宽和高
cv2.imshow('lena', lena)    # 显示原始图像

roi = lena[220: 400, 250: 350]

# 获取一个key，key是打码、解码用的密钥图像
key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
# ================获取打码脸=================
# Step 1：使用密钥key对原始图像lena加密
lenaXorKey = cv2.bitwise_xor(lena, key)
# Step 2：获取加密后图像的脸部区域（获取ROI）
secretFace = lenaXorKey[220: 400, 250: 350]
cv2.imshow('secretFace', secretFace)
# Step 3：划定ROI，其实没有实质性操作
# lena[220: 400, 250: 350]
# Step 4：将原始图像lena的脸部区域替换为加密后的脸部区域secretFace（ROI替换）
lena[220: 400, 250: 350] = secretFace
enFace = lena
cv2.imshow('enFace', enFace)
# ================脸部解码过程================
# Step 5：将脸部打码的图像enFace与密钥图像key进行异或运算，得到脸部的原始信息（按位异或运算）
extractOriginal = cv2.bitwise_xor(enFace, key)
# Step 6：获取解密后的图像的脸部区域（获取ROI）
face = extractOriginal[220: 400, 250: 350]
cv2.imshow('face', face)
# Step 7：将图像enFace的脸部区域替换为解密的脸部区域face（ROI替换）
enFace[220: 400, 250: 350] = face
deFace = enFace
cv2.imshow('deFace', deFace)
cv2.waitKey()
cv2.destroyAllWindows()
