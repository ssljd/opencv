import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==========构造提取感知哈希值函数============
def getHash(I):
    size = (8, 8)
    I = cv2.resize(I, size)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    m = np.mean(I)
    r = (I > m).astype(int)
    x = r.flatten()
    return x

# ==========构造计算汉明距离函数===========
def hamming(h1, h2):
    r = cv2.bitwise_xor(h1, h2)
    h = np.sum(r)
    return h

# ==========计算检索图像的感知哈希值============
o = cv2.imread('../img/apple.jpg')
h = getHash(o)
print('检索图像的感知哈希值为：\n', h)

# ==========计算指定文件夹下的所有图像感知哈希值============
images = []
EXTS = 'jpg', 'jpeg', 'gif', 'png', 'bmp'
for ext in EXTS:
    images.extend(glob.glob('../img/*.%s' % ext))
seq = []
for f in images:
    I = cv2.imread(f)
    seq.append((f, getHash(I)))

# ==========以图搜图核心：找出最相似图像============
# 计算检索图像与图像库内所有图像的距离，将最小距离对应的图像作为检索结果
distance = []
for x in seq:
    distance.append((hamming(h, x[1]), x[0]))

s = sorted(distance)

r1 = cv2.imread(str(s[0][1]))
r2 = cv2.imread(str(s[1][1]))
r3 = cv2.imread(str(s[2][1]))

# ==========绘制结果============
plt.figure('result')
plt.subplot(141), plt.imshow(cv2.cvtColor(o, cv2.COLOR_BGR2RGB)), plt.axis('off')
plt.subplot(142), plt.imshow(cv2.cvtColor(r1, cv2.COLOR_BGR2RGB)), plt.axis('off')
plt.subplot(143), plt.imshow(cv2.cvtColor(r2, cv2.COLOR_BGR2RGB)), plt.axis('off')
plt.subplot(144), plt.imshow(cv2.cvtColor(r3, cv2.COLOR_BGR2RGB)), plt.axis('off')
plt.show()