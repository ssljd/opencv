import cv2
import numpy as np
from scipy.spatial import distance as dist

# ========自定义函数：实现透视变换（倾斜矫正）========
# Step 1：参数fts是进行倾斜校正的轮廓的逼近多边形的四个定点
def myWarpPerspective(image, pts):
    # 确定4个顶点分别对应左上、右上、右下、左下4个顶点
    # Step 1.1：根据x轴坐标值对4个顶点进行排序
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # Step 1.2：将4个顶点划分为左侧两个、右侧两个
    left = xSorted[:2, :]
    right = xSorted[2:, :]
    # Step 1.3：在左侧寻找左上顶点、左下顶点
    # 根据y轴坐标值排序
    left = left[np.argsort(left[:, 1]), :]
    # 排在前面的是左上角顶点（tl:top-left）、排在后面的是左下角顶点（bl:bottom-left）
    (tl, bl) = left
    # Step 1.4：根据右侧两个顶点与左上角顶点的距离判断右侧两个顶点的位置
    # 计算右侧两个顶点距离左上角顶点的距离
    D = dist.cdist(tl[np.newaxis], right, 'euclidean')[0]
    # 右侧两个顶点中，距离左上角顶点远的点是右下角顶点（br），近的点是右上角顶点（tr）
    (br, tr) = right[np.argsort(D)[::-1], :]
    # Step 1.5：确定4个顶点分别对应左上、右上、右下、左下4个顶点中的哪个
    src = np.array([tl, tr, br, bl], dtype='float32')
    # =================以下5行为测试语句=================
    # srcx = np.array([tl, tr, br, bl], dtype='int32')
    # print('看看各个顶点在哪：\n', src)
    # test = image.copy()
    # cv2.polylines(test, [srcx], True, (255, 0, 0), 8)
    # cv2.imshow('image', test)
    # ========Step 2：根据pts的4个顶点，计算校正后的图像的宽度和高度=========
    # 校正后图像的大小计算比较随意，根据需要选用合适值即可
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 根据（左上，左下）和（右上，右下）的最大值，获取高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 根据宽度、高度，构造新图像dst对应的4个顶点
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype='float32')
    # print('看看目标如何：\n', dst)
    # 构造从src到dst的变换矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 完成从src到dst的透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # 返回透视变换的结果
    return warped

# 标准答案
ANSWER = {0: 0, 1: 3, 2: 1, 3: 0, 4: 2}
# 答案字典
answerDICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
# 读取原始图像
img = cv2.imread('../img/b.jpg')
# cv2.imshow('origin', img)
# 图像预处理：色彩空间变换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# 图像预处理：高斯滤波
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow('gaussian_blur', gaussian_blur)
# 图像预处理：边缘检测
edged = cv2.Canny(gaussian_blur, 50, 200)
# cv2.imshow('edged', edged)
# 查找轮廓
cts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContoure(img, cts, -1, (0, 0, 255), 3)
# 轮廓排序
list = sorted(cts, key=cv2.contourArea, reverse=True)
print('寻找轮廓的个数：', len(cts))
# cv2.imshow('drawContoure', img)
rightSum = 0
# 使用for循环，遍历每一个轮廓，找到答题卡的轮廓
# 对答题卡进行倾斜矫正处理
for c in list:
    peri = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, peri, True)
    print('顶点个数：', len(approx))
    # 4个顶点的轮廓是矩形
    if len(approx) == 4:
        # 对轮廓进行倾斜矫正，将其构造成一个矩形
        # 原始图像的倾斜校正用于后续标注
        paper = myWarpPerspective(img, approx.reshape(4, 2))
        # cv2.imshow('imgpaper', paper)
        # 对原始图像的灰度图像进行倾斜矫正，用于后续计算
        paperGray = myWarpPerspective(gray, approx.reshape(4, 2))
        # cv2.imshow('paper', paper)
        # cv2.imshow('paperGray', paperGray)
        # cv2.imwrite('paperGray.jpg', paperGray)
        # 反二值化预处理，将选项处理为白色，将答题卡整体背景处理黑色
        ret, thresh = cv2.threshold(paperGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow('thresh', thresh)
        # cv2.imwrite('thresh.jpg', thresh)
        # 在答题卡内寻找所有轮廓，此时找到所有轮廓
        # 既包含各个选项的轮廓，又包含答题卡内的说明文字等信息的轮廓
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 用options来保存每一个选项
        options = []
        # 遍历每一个轮廓cnts，将选项放入options
        # 依据条件
        # 条件1：轮廓如果宽度、高度都大于25像素
        # 条件2：纵横比介于[0.6, 1.3]
        # 若轮廓同时满足上述两个条件，则判定其为选项；否则，判定其为噪声
        for ci in cnts:
            # 获取轮廓的矩形包围框
            x, y, w, h = cv2.boundingRect(ci)
            # 计算纵横比
            ar = w / float(h)
            if w >= 25 and h >= 25 and ar >= 0.6 and ar <= 1.3:
                options.append(ci)
        # 将轮廓按位置关系自上而下存放
        boundingBoxes = [cv2.boundingRect(c) for c in options]
        (options, boundingBoxes) = zip(*sorted(zip(options, boundingBoxes), key=lambda b: b[1][1], reverse=False))
        for (tn, i) in enumerate(np.arange(0, len(options), 4)):
            boundingBoxes = [cv2.boundingRect(c) for c in options[i: i + 4]]
            (cnts, boundingBoxes) = zip(*sorted(zip(options[i: i+4], boundingBoxes), key=lambda b:b[1][0], reverse=False))
            # 构建列表ioptions，用来储存当前题目的每个选项
            ioptions = []
            for (ci, c) in enumerate(cnts):
                # 构造一个和答题卡同尺寸的掩模mask，灰度图像，黑色
                mask = np.zeros(paperGray.shape, dtype='uint8')
                # 在mask内，绘制当前遍历的选项
                cv2.drawContours(mask, [c], -1, 255, -1)
                # 使用按位与运算的掩膜模式，提取当前遍历的选项
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                # 计算当前遍历选项内像素值非0的轮廓个数
                total = cv2.countNonZero(mask)
                # 将选项像素值非0的轮廓个数、选项序号放入列表options内
                ioptions.append((total, ci))
            # 将每道题的4个选项按照像素值非0的轮廓的个数降序排序
            ioptions = sorted(ioptions, key=lambda x: x[0], reverse=True)
            # 获取包含最多白色像素点的选项索引
            choiceNum = ioptions[0][1]
            # 根据索引确定选项
            choice = answerDICT.get(choiceNum)
            if ANSWER.get(tn) == choiceNum:
                # 正确时，颜色为绿色
                color = (0, 255, 0)
                # 答对数量加1
                rightSum += 1
            else:
                # 错误时，颜色为红色
                color = (0, 0, 255)
            cv2.drawContours(paper, cnts[choiceNum], -1, color, 2)
        s1 = 'total: ' + str(len(ANSWER)) + ''
        s2 = 'right: ' + str(rightSum)
        s3 = 'score: ' + str(rightSum*1.0 / len(ANSWER) * 100) + ''
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(paper, s1 + ' ' + s2 + ' ' + s3, (10, 30), font, 0.5, (0, 0, 255), 2)
        cv2.imshow('score', paper)
        break
cv2.waitKey()
cv2.destroyAllWindows()