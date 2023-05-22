import cv2
import dlib
import numpy as np

# 读入图片
img = cv2.imread('../img/rotate.jpg')
# Step 1：初始化
# 构造人脸检测器
detector = dlib.get_frontal_face_detector()
# 检测人脸框
faceBoxs = detector(img, 1)
# 载入模型
predictor = dlib.shape_predictor('D:/Anaconda3/envs/airsim/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')
# Step 2：获取人脸集合
# 将Step 1获取的人脸框集合faceBoxs中的每个人脸框，逐个放入容器faces
faces = dlib.full_object_detections()   # 构造容器
for faceBox in faceBoxs:
    faces.append(predictor(img, faceBox))   # 把每个人脸框对应的放入容器faces
# Step 3：根据原始图像、人脸关键点获取人脸对齐结果
# 调用函数get_face_chips完成人脸图像的对齐（倾斜矫正）
faces = dlib.get_face_chips(img, faces, size=120)
# Step 4：将获取的每一张人脸显示出来
n = 0
for face in faces:
    n += 1
    face = np.array(face).astype(np.uint8)
    cv2.imshow('face%s'%(n), face)
cv2.imshow('original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()