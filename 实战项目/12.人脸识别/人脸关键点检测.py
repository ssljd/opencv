import cv2
import dlib
import numpy as np

# 读入图片
img = cv2.imread('../img/y.png')
# Step 1：初始化
# 构造人脸检测器
detector = dlib.get_frontal_face_detector()
# Step 2：检测人脸
faces = detector(img, 0)
# Step 3：载入模型
predictor = dlib.shape_predictor('D:/Anaconda3/envs/airsim/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')
# Step 4：获取每一张图像的关键点
for face in faces:
    # 获取关键点
    shape = predictor(img, face)
    # Step 5：绘制每一张人脸的关键点
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    for idx, point in enumerate(landmarks):
        # 当前关键点的坐标
        pos = (point[0, 0], point[0, 1])
        # 针对当前关键点绘制一个实心圆
        cv2.circle(img, pos, 2, color=(0, 255, 0), thickness=-1)
        # 字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx + 1), pos, font, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
cv2.imshow('original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()