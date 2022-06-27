import cv2

img = cv2.imread("../img/1.png")
res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# height,width = img.shape[:2]
# res = cv2.resize(img,(2 * width,2 * height),interpolation=cv2.INTER_CUBIC)
while 1:
    cv2.imshow("image", img)
    cv2.imshow("res", res)
    if cv2.waitKey(1) & 0xFF == 27:  # 如果按键是esc,则关闭窗口
        break
cv2.destroyAllWindows()