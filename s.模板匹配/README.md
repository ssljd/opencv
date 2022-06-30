# 模板匹配
- template_matching.py [code](template_matching.py)
- mul_template_matching.py [code](mul_template_matching.py)
## 单对象的模板匹配
```
cv2.matchTemplate(image, temp1, methord)
```
1）image：输入图像（包含要检测的对象图像）
2）temp1：对象图像
3）method：模板匹配的方法
- 方法1：cv2.TM_SQDIFF
> R(x,y) = $\sum_{x',y'}$(T(x',y')- I(x + x', y + y'))$^2$
- 方法2：cv2.TM_SQDIFF_NORMED
> $$R(x,y) = \frac{\sum_{x',y'}(T(x',y') - I(x+x',y+y'))^2}{\sqrt{\sum_{x',y'}T(x',y')^2\centerdot \sum_{x',y'}I(x+x',y+y')^2}} $$
- 方法3：cv2.TM_CCORR
> $$R(x,y) = \sum_{x',y'}(T(x',y') \centerdot I(x+x',y+y')) $$
- 方法4：cv2.TM_CCORR_NORMED
> $$R(x,y) = \frac{\sum_{x',y'}(T(x',y') \centerdot I(x+x',y+y'))}{\sqrt{\sum_{x',y'}T(x',y')^2 \centerdot \sum_{x',y'}I(x+x',y+y')^2}} $$
- 方法5：cv2.TM_CCOEFF
> $$R(x,y) = \sum_{x',y'}(T'(x',y') \centerdot I'(x+x',y+y'))$$
- 方法6：cv2.TM_CCOEFF_NORMED
> $$R(x,y) = \frac{\sum_{x',y'}(T'(x',y') \centerdot I'(x+x',y+y'))}{\sqrt{\sum_{x',y'}T'(x',y')^2 \centerdot \sum_{x',y'}I'(x+x',y+y')^2}}$$
## 获取最佳匹配结果
```
cv2.minMaxLoc(src, mask=None)
```
1）src：一个矩阵
- 函数功能：假设有一个矩阵a,现在需要求这个矩阵的最小值，最大值，并得到最大值，最小值的索引。咋一看感觉很复杂，但使用这个cv2.minMaxLoc()函数就可全部解决。函数返回的四个值就是上述所要得到的。

