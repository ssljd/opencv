# ==============导入库=================
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# 获取背景信息
ret, back = cap.read()
# 实时采集
while(cap.isOpened()):
    # 实时采集摄像头信息
    ret, fore = cap.read()
    # 没有捕捉到任何信息，中断
    if not ret:
        break
    # 实时显示采集到的摄像头视频信息
    cv2.imshow('fore', fore)
    # 色彩空间转换，由BGR色彩空间至HSV色彩空间
    hsv = cv2.cvtColor(fore, cv2.COLOR_BGR2HSV)
    # 红色区间1
    redLower = np.array([0, 120, 70])
    redUpper = np.array([10, 255, 255])
    # 红色在HSV色彩空间内的范围1
    maska = cv2.inRange(hsv, redLower, redUpper)
    # 红色区间2
    redLower = np.array([170, 120, 70])
    redUpper = np.array([180, 255, 255])
    # 红色在HSV色彩空间内的范围2
    maskb = cv2.inRange(hsv, redLower, redUpper)
    # 红色整体区间 = 红色区间1 + 红色区间2
    mask1 = maska + maskb
    # 膨胀
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    # 按位取反
    mask2 = cv2.bitwise_not(mask1)
    # 提取back中mask1指定的范围
    result1 = cv2.bitwise_and(back, back, mask=mask1)
    # 提取fore中mask2指定的范围
    result2 = cv2.bitwise_and(fore, fore, mask=mask2)
    result = result1 + result2
    # 显示最终结果
    cv2.imshow('result', result)
    k = cv2.waitKey(1000)
    if k == 27:
        break
    cv2.destroyAllWindows()
