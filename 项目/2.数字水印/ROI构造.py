import cv2
import numpy as np

# 嵌入过程
def embed(O, W):
    # Step 1：最低有效位清0
    OBZ = O - O % 2
    # Step 2：将水印图像处理为二值形式
    WB = (W / 255).astype(np.uint8)
    '''更严谨的方式'''
    # W[W>127] = 255
    # WB = W
    # Step 3：嵌入水印图像
    OW = OBZ + WB
    # 显示原始图像、水印图像、嵌入水印的图像
    cv2.imshow('Original', O)
    cv2.imshow('watermark', W)
    cv2.imshow('embed', OW)
    return OW

def extract(OW):
    # Step 4：获取水印图像OW的最低有效位，获取数字水印信息
    EWB = OW % 2
    # Step 5：将二值形式的水印图像的数值1乘以255，得到256级灰度值图像
    # 将前景色由黑色（对应数值1）变为白色（对应数值255）
    EW = EWB * 255
    # 显示提取结果
    cv2.imshow('extractedWatermark', EW)

if __name__ == '__main__':
    # 读取原始载体图像O
    O = cv2.imread('../img/lena.bmp', 0)
    # 读取水印图像W
    W = cv2.imread('../img/watermark.bmp', 0)
    # 嵌入水印图像
    OW = embed(O, W)
    extract(OW)
    # 显示控制、释放窗口
    cv2.waitKey()
    cv2.destroyAllWindows()