import cv2
import numpy as np

lena = cv2.imread('../img/lena.bmp', 0)  # 读取图像
r, c = lena.shape   # 获取图像的宽和高
key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)    # 密钥图像
encryption = cv2.bitwise_xor(lena, key)     # 加密图像
decryption = cv2.bitwise_xor(encryption, key)   # 揭秘图像
cv2.imshow('lena', lena)
cv2.imshow('key', key)
cv2.imshow('encryption', encryption)
cv2.imshow('decryption', decryption)
cv2.waitKey()
cv2.destroyAllWindows()