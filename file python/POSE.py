import cv2 
from ultralytics import YOLO
import numpy as np
import os
import mediapipe as mp

class Pose():
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
    def find_pose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def get_point(self,img):        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        c_lm = []
        list_no = [0,1,3,4,6,7,8]
        ok=True
        if self.results:
            if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    if id not in list_no:
                        c_lm.append(lm.x)
                        c_lm.append(lm.y)
                        c_lm.append(lm.z)
                        c_lm.append(lm.visibility)
                return c_lm,True
            else:
                return None,False
        else:
            return None,False
    def draw_landmark_on_image(self,img):
        self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx,cy),5,(0,0,255),cv2.FILLED)
        return img

    




