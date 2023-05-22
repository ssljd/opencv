import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# ============主程序====================
while(cap.isOpened()):
    ret, frame = cap.read()     # 读取摄像头图像
    frame = cv2.flip(frame, 1)  # 绕着y轴方向反转图像
    # ================设定一个固定区域作为识别区域==============
    roi = frame[10:410, 200:600]    # 将右上角设定为识别区域
    cv2.rectangle(frame, (200, 10), (600, 410), (0, 0, 255), 0)     # 将选定的区域标记出来
    # ===============在hsv色彩空间内检测出皮肤==================
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 色彩空间转换
    lower_skin = np.array([0, 28, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # ================预处理====================
    kernel = np.ones((2, 2), np.uint8)  # 构造一个核
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    # ================找出轮廓====================
    # 查找所有轮廓
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 从所有轮廓中找到最大轮廓，并将其作为手势轮廓
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    areacnt = cv2.contourArea(cnt)
    # ================获取轮廓的凸包====================
    hull = cv2.convexHull(cnt)  # 获取轮廓的凸包，用于计算面积，返回坐标值
    areahull = cv2.contourArea(hull)    # 获取凸包的面积
    # ================获取轮廓面积、凸包的面积比====================
    arearadio = areacnt / areahull
    # ================获取凸缺陷====================
    hull = cv2.convexHull(cnt, returnPoints=False)  # 使用索引
    defects = cv2.convexityDefects(cnt, hull)   # 获取凸缺陷
    # ================凸缺陷处理====================
    n = 0
    for i in range(defects.shape[0]):
        s, e, f, d, = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        # ================计算手指之间的角度====================
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        # ================绘制手指间的凸包最远点====================
        if angle<90 and d>20:
            n += 1
            cv2.circle(roi, far, 3, [255, 0, 0], -1)
        # ================绘制手势的凸包====================
        cv2.line(roi, start, end, [0, 255, 0], 2)
    # ================通过凸缺陷个数及凸缺陷和凸包的面积比判断识别结果====================
    if n == 0:                  # 0个凸缺陷，手势可能表示数值0，也可能表示数值1
        if arearadio > 0.9:     # 轮廓面积/凸包面积>0.9，判定为拳头识别手势为数值0
            result = '0'
        else:
            result = '1'        # 轮廓面积/凸包面积<0.9，判定为拳头识别手势为数值0
    elif n == 1:                # 1个凸缺陷，对应2根手指，识别手势为数值2
        result = '2'
    elif n == 2:                # 2个凸缺陷，对应3根手指，识别手势为数值3
        result = '3'
    elif n == 3:                # 3个凸缺陷，对应4根手指，识别手势为数值4
        result = '4'
    elif n == 4:                # 4个凸缺陷，对应5根手指，识别手势为数值5
        result = '5'
    # ================设置与显示识别结果相关的参数====================
    org = (400, 80)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 0, 255)
    thichnesss = 3
    # ================显示识别结果====================
    cv2.putText(frame, result, org, font, fontScale, color, thichnesss)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
cv2.waitKey()
cv2.destroyAllWindows()