import cv2
import numpy as np

img = cv2.imread('./img/harris.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转换为GRAY格式
gray = np.float32(gray)
#输入图像必须是float32，最后一个参数在0.04到0.05之间
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#结果是扩大标记的角落，不重要
# 找到Harris角
dst = cv2.dilate(dst, None)  # 扩大标记的角落
ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
dst = np.uint8(dst)
# 查找质心
# connectedComponentsWithStats（InputArray映像、OutputArray标签、OutputArray统计信息、，
# OutputArray质心，int connectivity=8，int ltype=）
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# cv2.cornerSubPix（图像、角点、winSize、zeroZone、条件）在搜索区域中部死亡区域的一半大小
# 下面的公式中的求和没有完成。有时也会用到
# 避免自相关矩阵可能出现的奇异性。值（-1，-1）
# 表示没有这样的大小。
# 返回值由角点坐标组成的一个数组（而非图像）
corners = cv2.cornerSubPix(gray, np.float32(centroids),(5,5),(-1,-1),criteria)
print(corners)
corners.sort()
outfile = open('corners.txt', 'w')
for i in range(len(corners)):
    outfile.write(str(corners[i][0]) + ' ' + str(corners[i][1]) + '\n')
outfile.close()


img[dst > 0.01 * dst.max()] = [0, 60, 255]#控制角点的颜色
cv2.imwrite("finalimage.png", img)
# cv2.imshow("dst", img)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
