import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../img/IMG_20201027_162850.jpg",0)
#flatten()将数组变成一堆
hist,bins = np.histogram(img.flatten(),256,[0,256])
#计算累积分布图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
#构建Numpy掩膜数组，cdf为数组，当数组元素为0时，掩盖
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
#对被掩盖的元素赋值，这里赋值为0
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img1 = cdf[img]
plt.plot(cdf_normalized,color='b')
plt.hist(img1.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc='upper left')
plt.show()

img = cv2.imread('../img/IMG_20201027_162850.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
#创建一个CLAHE对象
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(img)
while(1):
    cv2.imshow("cl1", cl1)
    cv2.imshow("res", res)  # 并排堆叠图像
    cv2.imwrite('clahe.jpg',cl1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()