import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import LC
import POSE


labels = ['binh thuong','nguy hiem','nguy hiem cao']
n_time_steps = 15
lm_list = []
lc = LC.reg_lc()
pose = POSE.Pose()
color = (0,0,0)
color1 = (255,144,30)
color2 = (0,215,255)
color3 = (0,0,255)
name_label = "..."
cnt = 3

model = tf.keras.models.load_model("C:/Users/Admin/OneDrive/Documents/AI/model1v4.h5")
path = 'C:/Users/Admin/OneDrive/Documents/newdata/video_test.mp4'

cap = cv2.VideoCapture(path)

i = 0
list_box = np.array([])

while True:

    success, img = cap.read()
    ok,box,boxn = lc.find_lc(img)
    if ok:
        cv2.rectangle(img,box,(0,255,0),2)
        pose.find_pose(img, True)
        list_pose,ok2 = pose.get_point(img)
        if ok2:
            i+=1
            itembox = np.concatenate((list_pose, boxn),dtype = float)
            list_box = np.append(list_box,itembox)
         
            if i == n_time_steps:
                list_box = np.reshape(list_box,(15,108))
                list_box = np.expand_dims(list_box, axis=0)
                print(list_box.shape)
                results = model.predict(list_box)
               
                if labels[results.argmax()]=='binh thuong':
                    color = color1
                    name_label = 'NORMAL'
                if labels[results.argmax()]=='nguy hiem':
                    color = color2
                    name_label = 'WARNING'
                if labels[results.argmax()]=='nguy hiem cao':
                    color = color3
                    name_label = 'DANGEROUS'
                i=0
                list_box = np.array([])
    cnt-=1
    if cnt == 0:
        cv2.putText(img, f"{name_label}",(20,50), cv2.FONT_HERSHEY_PLAIN, 4,color, 3)         
        cnt=3
                

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
