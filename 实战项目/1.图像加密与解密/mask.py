import cv2
import numpy as np

lena = cv2.imread('../img/lena.bmp', 0)  # 读取图像
r, c = lena.shape   # 获取图像的宽和高
cv2.imshow('lena', lena)    # 显示原始图像

mask = np.zeros((r, c), dtype=np.uint8)
mask[220: 400, 250: 350] = 1

# 获取一个key，key是打码、解码用的密钥图像
key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
# ================获取打码脸=================
# Step 1：使用密钥key对原始图像lena加密
lenaXorKey = cv2.bitwise_xor(lena, key)
# Step 2：获取加密图像的脸部信息encryptFace
encryptFace = cv2.bitwise_and(lenaXorKey, mask * 255)
# Step 3：将图像lena内的脸部区域的像素值设置为0，得到图像noFacel
noFacel = cv2.bitwise_and(lena, (1 - mask) * 255)
# Step 4：得到打码的lena图像
maskFace = encryptFace + noFacel
cv2.imshow('maskFace', maskFace)
# ================将打码脸解码=================
# Step 5：将脸部打码的lena图像与密钥图像key进行异或运算，得到脸部的原始信息
extractOriginal = cv2.bitwise_xor(maskFace, key)
# Step 6：将解码的脸部信息extractOriginal提取出来，得到图像encryptFace
encryptFace = cv2.bitwise_and(extractOriginal, mask * 255)
# Step 7：从脸部打码的lena图像内提取没有脸部信息的lena图像，得到图像noFace2
noFace2 = cv2.bitwise_and(maskFace, (1 - mask) * 255)
# Step 8：得到解码图像extractLena
extractLena = noFace2 + encryptFace
cv2.imshow('extractLena', extractLena)
cv2.waitKey()
cv2.destroyAllWindows()
