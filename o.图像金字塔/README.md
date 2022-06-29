# 图像金字塔
## 下采样
```
cv2.pyrDown(src_img, dstsize)
```
1）src_img：原始图像  
2）dssize：目标图像大小；默认行和列都会变成原始图像行和列的1/2，整幅图像会变成原始图像的1/4
## 上采样
```
cv2.pyrUp(src_img, dstsize)
```
1）src_img：原始图像  
2）dssize：目标图像大小；默认行和列都会变成原始图像行和列的2，整幅图像会变成原始图像的4