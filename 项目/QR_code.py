import cv2
import numpy as np

# red代表红色，green代表绿色，blue代表蓝色
def judge_message(codeinfo):
    message = []
    if codeinfo[:2] == "红色":
        message.append("red")
    elif codeinfo[:2] == "绿色":
        message.append("green")
    elif codeinfo[:2] == "蓝色":
        message.append("blue")
    if codeinfo[2:] == "红色":
        message.append("red")
    elif codeinfo[2:] == "绿色":
        message.append("green")
    elif codeinfo[2:] == "蓝色":
        message.append("blue")
    return message

global codeinfo
# 实时视频二维码检测
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # 读取二维码
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 设置检测器
    qrcoder = cv2.QRCodeDetector()
    # 检测识别二维码
    codeinfo, points, straight_qrcode = qrcoder.detectAndDecode(gray)
    result = np.copy(frame)
    # 输出识别二维码的信息
    print("qrcode information is : \n%s" % codeinfo)
    cv2.imshow("result", result)
    flag = cv2.waitKey(1)
    if len(codeinfo) == 4:
        break
cap.release()
cv2.destroyAllWindows()
# ****************************************************************** #
# 读取二维码
# src = cv2.imread("qrcode.png")
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # 设置检测器
# qrcoder = cv2.QRCodeDetector()
# # 检测识别二维码
# codeinfo, points, straight_qrcode = qrcoder.detectAndDecode(gray)
# result = np.copy(src)
# cv2.drawContours(result, [np.int32(points)], 0, (0, 0, 255), 2)
# # 输出识别二维码的信息
# print("qrcode information is : \n%s" % codeinfo)

# # 显示图片
# cv2.imshow("result", result)
# # cv2.imshow("qrcode roi", np.uint8(straight_qrcode))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# **************************************************************** #

# 颜色阈值列表
color_list = {
    "blue": {"Lower": np.array([0, 60, 60]), "Upper": np.array([30, 255, 255])},
    "red": {"Lower": np.array([110, 80, 46]), "Upper": np.array([134, 255, 255])},
    "green": {"Lower": np.array([35, 50, 35]), "Upper": np.array([60, 255, 255])},
    "white": {"Lower": np.array([0, 0, 240]), "Upper": np.array([255, 15, 255])},
    "yellow": {"Lower": np.array([30, 30, 45]), "Upper": np.array([255, 77, 255])},
}

messages = judge_message(codeinfo)
print(messages)

for message in messages:
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
                    color_list[message]["Lower"],
                    color_list[message]["Upper"],)
                # 腐蚀操作
                inRange_hsv1 = cv2.erode(inRange_hsv1, None, iterations=2)
                # 膨胀操作，先腐蚀后膨胀以滤除噪声
                inRange_hsv1 = cv2.dilate(inRange_hsv1, None, iterations=2)
                cnts1 = cv2.findContours(
                    inRange_hsv1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                cv2.imshow("mask", inRange_hsv1)
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
                    # x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度

                    # 只处理尺寸最大的轮廓
                    if radius > 20:

                        print(CornerNum)
                        # 轮廓对象分类
                        if CornerNum == 8:
                            objType = "Circle"
                        elif CornerNum == 7:
                            objType = "Square"
                        elif CornerNum == 6:
                            objType = "Hexagon"
                        else:
                            objType = "N"

                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制边界框
                        # cv2.putText(frame, objType, (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        #             (0, 0, 0),
                        #             1)  # 绘制文字

                        # 画出最小外接圆
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        # 画出重心
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        print("中心坐标：", x, y)

            # c1 = max(cnts1, key=cv2.contourArea)
            # # c2 = max(cnts2, key=cv2.contourArea)
            # rect1 = cv2.minAreaRect(c1)
            # # rect2 = cv2.minAreaRect(c2)
            # box1 = cv2.boxPoints(rect1)
            # # box2 = cv2.boxPoints(rect2)
            #
            # cv2.drawContours(frame, [np.int0(box1)], -1, (0, 255, 255), 2)
            # cv2.drawContours(frame, [np.int0(box2)], -1, (0, 255, 255), 2)
                cv2.imshow("camera", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()
