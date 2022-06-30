import cv2

cv2.VideoCapture.open()
X = 1
count = 1
def test():
    while True:
        # 调用摄像头
        cam = cv2.VideoCapture(2)
        while True:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            cv2.waitKey(1)
            # 加一个鼠标点击事件，frame传给了OnMouseAction的param
            # 鼠标左键点击相框，相当于按下快门
            cv2.setMouseCallback("test", OnMouseAction, frame)
            global X
            if X == 2:
                X = 1
                break
            if cv2.waitKey(1) & 0xFF == 27:  # 如果按下esc键，则跳出函数，停止拍照
                return
        cam.release()
        cv2.destroyAllWindows()
def OnMouseAction(event,x,y,flags,param):
    #cv2.EVENT_LBUTTONDOMN左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        global count
        cv2.imwrite(f"{count}.png",param) #保存图片
        count += 1
        global X
        X = 2
test()