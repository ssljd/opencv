import cv2
import numpy as np

# 颜色阈值列表
color_list = {
    "blue": {"Lower": np.array([0, 60, 60]), "Upper": np.array([30, 255, 255])},
    "red": {"Lower": np.array([110, 80, 46]), "Upper": np.array([134, 255, 255])},
    "green": {"Lower": np.array([35, 50, 35]), "Upper": np.array([60, 255, 255])},
    "white": {"Lower": np.array([0, 0, 145]), "Upper": np.array([100, 100, 255])},
}

cap = cv2.VideoCapture(0)
cv2.namedWindow("camera", cv2.WINDOW_AUTOSIZE)
while cap.isOpened():  # 判断摄像头是否打开
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_RGB2HSV)  # 转化为HSV图像
            erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀，粗的变细
            # 对图像进行二值化处理
            inRange_hsv1 = cv2.inRange(
                erode_hsv.copy(),
                color_list["red"]["Lower"],
                color_list["red"]["Upper"],
            )
            # 腐蚀操作
            inRange_hsv1 = cv2.erode(inRange_hsv1, None, iterations=2)
            # 膨胀操作，先腐蚀后膨胀以滤除噪声
            inRange_hsv1 = cv2.dilate(inRange_hsv1, None, iterations=2)
            cnts1 = cv2.findContours(
                inRange_hsv1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2]
            # cv2.imshow("mask", inRange_hsv1)
            if len(cnts1) > 0:
                # 找到面积最大的轮廓
                c = max(cnts1, key=cv2.contourArea)
                # 使用最小外接圆圈出面积最大的轮廓
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # 计算轮廓的矩
                M = cv2.moments(c)
                # 计算轮廓的重心
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                perimeter = cv2.arcLength(c, True)  # 计算轮廓周长
                approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)  # 获取轮廓角点坐标
                CornerNum = len(approx)  # 轮廓角点的数量
                # 只处理尺寸最大的轮廓
                if radius > 20 and y > 150 and y < 310:
                    # 画出最小外接圆
                    cv2.circle(
                        frame, (int(x), int(y)), int(radius), (0, 255, 255), 2
                    )
                    # 画出重心
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    print("中心坐标：", x, y)
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
cap.release()
cv2.destroyAllWindows()