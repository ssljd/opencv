# ==========导入库============
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

# ==========使用字典表示模板、部分省份简称============
templateDict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J',
                19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T',
                28: 'U', 29: 'V', 30: 'W', 31: 'X', 32: 'Y', 33: 'Z', 34: '京', 35: '津', 36: '冀',
                37: '晋', 38: '蒙', 39: '辽', 40: '吉', 41: '黑', 42: '沪', 43: '苏', 44: '浙', 45: '皖',
                46: '闽', 47: '赣', 48: '鲁', 49: '豫', 50: '鄂', 51: '湘', 52: '粤', 53: '桂', 54: '琼',
                55: '渝', 56: '川', 57: '贵', 58: '云', 59: '藏', 60: '陕', 61: '甘', 62: '青', 63: '宁',
                64: '新', 65: '港', 66: '澳', 67: '台',}

# ==========提取车牌函数============
def getPlate(image):
    rawImage = image.copy()
    # 去噪处理
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # 色彩空间转换（BGR->GRAY）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Sobel算子（x轴方向边缘梯度）
    Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX
    # 阈值处理
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # 闭运算：先膨胀后腐蚀，车牌各个字符是分散的，让车牌构成一个整体
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
    # 开运算，先腐蚀后膨胀，去除噪声
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelY)
    # 中值滤波：去除噪声
    image = cv2.medianBlur(image, 15)
    # 查找轮廓
    contours, w1 = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # 测试语句
    # image = cv2.drawContours(rawImage.copy(), contours, -1, (0, 0, 255), 3)
    # cv2.imshow('imagecc', image)

    # 逐个遍历轮廓，将宽高比大于3的轮廓确定为车牌
    s = 0
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        size = item.size
        if size > s:
            plate = rawImage[y: y + height, x: x + weight]
            s = size

    return plate

# ==========图像预处理函数，图像去噪等处理============
def preprocessor(image):
    # 图像去噪和灰度处理
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # 色彩空间转换
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 二值化
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    # 膨胀处理，让一个字构成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)
    return image

# ==========拆分车牌函数，使车牌内各个字符分离============
def splitPlate(image):
    # 查找轮廓，各个字符的轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    # 遍历所有轮廓
    for item in contours:
        rect = cv2.boundingRect(item)
        words.append(rect)
    # 按照x轴坐标值排序
    words = sorted(words, key=lambda s: s[0], reverse=False)
    # 用word存放左上角起始点及长宽值
    plateChars = []
    for word in words:
        # 筛选字符的轮廓
        if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 8)) and (word[2] > 3):
            plateChar = image[word[1]: word[1] + word[3], word[0]: word[0] + word[2]]
            plateChars.append(plateChar)

    return plateChars

# ==========获取所有模板图像的文件名============
def getcharacters():
    c = []
    for i in range(0, 67):
        words = []
        words.extend(glob.glob('template/' + templateDict.get(i) + '*.*'))
        c.append(words)
    return c

# ==========计算匹配值函数============
def getMatchValue(template, image):
    # 读取模板图像
    templateImage = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)
    # 色彩空间转换
    templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)
    # 模板图像阈值处理，灰度->二值
    ret, templateImage = cv2.threshold(templateImage, 0, 255, cv2.THRESH_OTSU)
    # 获取待识别图像尺寸
    height, width = image.shape
    # 将模板图像尺寸调整为待识别图像尺寸
    templateImage = cv2.resize(templateImage, (width, height))
    # 计算模板图像、待识别图像的模板匹配值
    result = cv2.matchTemplate(image, templateImage, cv2.TM_CCOEFF)

    return result[0][0]

def matchChars(plates, chars):
    # 储存所有识别结果
    results = []
    # 逐个遍历待识别字符
    for platechar in plates:
        bestMatch = []
        # 中间层循环：遍历所有特征字符
        for words in chars:
            match = []
            # 最内层循环：遍历每一个特征字符的所有模板
            for word in words:
                result = getMatchValue(word, platechar)
                match.append(result)
            if match == []:
                continue
            bestMatch.append(max(match))        # 将每个字符模板的最佳匹配模板加入bestMatch
        i = bestMatch.index(max(bestMatch))     # i是最佳匹配的字符模板的索引
        r = templateDict[i]                     # r是单个待识别字符的识别结果
        results.append(r)                       # 将每一个分割字符的识别结果加入results

    return results

if __name__ == '__main__':
    # 读取原始图像
    image = cv2.imread('../img/gua.jpg')
    # 显示原始图像
    cv2.imshow('original', image)
    # 获取车牌
    image = getPlate(image)
    cv2.imshow('plate', image)
    # 预处理
    image = preprocessor(image)
    # 分割车牌，将每个字符独立出来
    plateChars = splitPlate(image)
    # 逐个遍历字符
    for i, im in enumerate(plateChars):
        cv2.imshow('plateChars' + str(i), im)
    # 获取所有模板文件
    chars = getcharacters()
    # 使用模板chars逐个识别字符集plates
    results = matchChars(plateChars, chars)
    # 将列表转化为字符串
    results = ''.join(results)
    print('识别结果为：', results)
    cv2.waitKey()
    cv2.destroyAllWindows()