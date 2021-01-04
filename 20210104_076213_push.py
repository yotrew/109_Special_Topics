from flask import Flask, request, abort
from linebot import ( LineBotApi, WebhookHandler
)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage ,ImageMessage,)

#from datetime import datetime

import sys,os,dlib,glob,numpy
from skimage import io
import cv2
#import imutils
import time


# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 比對人臉圖片資料夾名稱
faces_folder_path = "./rec"

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()

# 載入人臉特徵點檢測器
sp = dlib.shape_predictor(predictor_path)

# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
candidate = []

# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    base = os.path.basename(f)
    # 依序取得圖片檔案人名
    candidate.append(os.path.splitext(base)[0])
    img = io.imread(f)

    # 1.人臉偵測
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        # 2.特徵點偵測
        shape = sp(img, d)
 
        # 3.取得描述子，128維特徵向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        # 轉換numpy array格式
        v = numpy.array(face_descriptor)
        descriptors.append(v)
if len(descriptors) != len(candidate):
    print("某張影像無法偵測出人臉!!!!")
    print("請確認後再執行!!!")
    print("共有:",len(candidate),"張,只辨識出:",len(descriptors),"張")
    sys.exit(0)#若所有的影像無法完整辨識出,再跳出程式,由人工刪除檔案
print("共有:",len(candidate),"張,辨識出:",len(descriptors),"張")
    
line_bot_api= LineBotApi('sOl9LdHHdqNYBQlEJEPJXiKs6EFma4kDu0K2wUMsmWwoSaMN24i8wdkI3tyjxhACcuM8wOn8eEGckqjR4NqkK4cONJAEXOCDjG3B4CPtg15poVj7Ypybq180JCeDLsscOz8xW1+sc6MeJb2gdXsdTwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('0613d6b2f8c86f5311ab0a67dce88c50')
To_user_ID="Uf20fb51315cb8d1d14f072ff10688529"



rec_pre_time=0
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(cv2. CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 480)
    
    #當攝影機打開時，對每個frame進行偵測
    while(cap.isOpened()):
        #讀出frame資訊
        ret, frame = cap.read()
    
        # 針對需要辨識的人臉同樣進行處理
        dets = detector(frame, 1)
    
        dist = []
        for k, d in enumerate(dets):
            shape = sp(frame, d)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            d_test = numpy.array(face_descriptor) # last face
    
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            # 以方框標示偵測的人臉
            cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2.LINE_AA)
     
            # 計算歐式距離
            for i in descriptors:
                dist_ = numpy.linalg.norm(i -d_test)
                dist.append(dist_)
    
        # 將比對人名和比對出來的歐式距離組成一個dict
        c_d = dict( zip(candidate,dist))
    
        # 根據歐式距離由小到大排序
        cd_sorted = sorted(c_d.items(), key = lambda d:d[1])
    
        print(cd_sorted)
        print(len(cd_sorted))
        if(len(cd_sorted)<=0):
            continue
        # 取得最短距離就為辨識出的人名
        rec_name = cd_sorted[0][0]
    
        # 將辨識出的人名印到圖片上面
        cv2.putText(frame, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2.LINE_AA)
        if(rec_name=="yotrew" ):
            rec_appeared_time=time.time()
            print(time.time(),rec_appeared_time-rec_pre_time)
            if((rec_appeared_time-rec_pre_time)>60):
                print("Line...")
                rec_pre_time=rec_appeared_time
                line_bot_api.push_message(To_user_ID, TextSendMessage(text='偵測到老蘇來了~~塊桃啊!!'))
        cv2.imshow("Face Recognition", frame)
    
        #如果按下ESC键，就退出
        if cv2.waitKey( 10) == 27:
            break
    
    
    #釋放記憶體
    cap.release()
    #關閉所有視窗
    cv2.destroyAllWindows()