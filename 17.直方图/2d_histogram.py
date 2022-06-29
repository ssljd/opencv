import cv2
import sys
import video
import numpy as np
from time import clock
from matplotlib import pyplot as plt

img = cv2.imread("../img/IMG_20201027_162850.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
plt.imshow(hist,interpolation='nearest')
plt.show()

if __name__ == '__main__':
    #构建HSV颜色地图
    hsv_map = np.zeros((180,256,3),np.uint8)
    # np.indices 可以返回由数组索引构建的新数组。
    # 例如：np.indices（（3,2））；其中（3,2）为原来数组的维度：行和列。
    # 返回值首先看输入的参数有几维：（3,2）有 2 维，所以从输出的结果应该是
    # [[a],[b]], 其中包含两个 3 行，2 列数组。
    # 第二看每一维的大小，第一维为 3, 所以 a 中的值就 0 到 2（最大索引数），
    # a 中的每一个值就是它的行索引；同样的方法得到 b（列索引）
    # 结果就是
    # array([[[0, 0],
    # [1, 1],
    # [2, 2]],
    #
    # [[0, 1],
    # [0, 1],
    # [0, 1]]])
    h,s = np.indices(hsv_map.shape[:2])
    hsv_map[:,:,0] = h
    hsv_map[:,:,1] = s
    hsv_map[:,:,2] = 255
    hsv_map = cv2.cvtColor(hsv_map,cv2.COLOR_HSV2BGR)
    cv2.imshow('hist',hsv_map)
    cv2.waitKey(0)
    cv2.namedWindow('hist',0)
    hist_scale = 10
    def set_scale(val):
        global hist_scale
        hist_scale = val
    cv2.createTrackbar('scale','hist',hist_scale,32,set_scale)
    try:fn = sys.argv[1]
    except:fn = 0
    cam = video.create_capture()