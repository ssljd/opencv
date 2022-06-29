import cv2

'''用摄像头捕获视频'''
cap = cv2.VideoCapture(0)   # 打开摄像头
while(cap.isOpened()):
    rat, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
cap.release()
cv2.destroyAllWindows()

'''从文件中播放视频'''
cap = cv2.VideoCapture(0)   # 打开摄像头
cap.isOpened()  # 检查是否成功初始化摄像头设备
# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')    # FourCC编码，具体是什么还不知道
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
while(cap.isOpened()):
    rat, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame",gray)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
cap.release()
cv2.destroyAllWindows()

'''保存视频'''
cap = cv2.VideoCapture(1)   # 打开摄像头
cap.isOpened()  # 检查是否成功初始化摄像头设备
# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')    # FourCC编码，具体是什么还不知道
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (630, 480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,0)
        # 写翻转的帧
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("frame",gray)
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()