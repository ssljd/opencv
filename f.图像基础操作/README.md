# 图像基础操作

### 填充
```
cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
```
- src：输入图像
- top, bottom, left, right：对应边界的像素数目
- borderType：要添加那种类型的边界
  - cv2.BORDER_CONSTANT：添加有颜色的常数值边界
  - cv2.BORDER_REFLECT：边界元素的镜像，例子： fedcba|abcde-fgh|hgfedcb
  - cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT：例子： gfedcb|abcdefgh|gfedcba
  - cv2.BORDER_REPLICATE：重复最后一个元素
- value：边界颜色