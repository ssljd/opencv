import cv2

def detect(image, winStride, padding, scale, useMeanshiftGrouping):
    hog = cv2.HOGDescriptor()   # 初始化HOG描述符
    # 设置SVM为一个预训练好的行人检测器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # 获取行人对应的矩形框及对应的权重值
    (rects, weights) = hog.detectMultiScale(image,
                                            winStride=winStride,
                                            padding=padding,
                                            scale=scale,
                                            useMeanshiftGrouping=useMeanshiftGrouping)
    # 绘制每一个矩形框
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('result', image)     # 显示原始结果

if __name__ == '__main__':
    image = cv2.imread('../img/people.jpg')
    winStride = (8, 8)
    padding = (2, 2)
    scale = 1.03
    useMeanshiftGrouping = True
    detect(image, winStride, padding, scale, useMeanshiftGrouping)
    cv2.waitKey(0)
    cv2.destroyAllWindows()