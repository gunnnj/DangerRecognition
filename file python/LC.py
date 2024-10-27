import cv2 
from ultralytics import YOLO
import numpy as np
import os
import mediapipe

class reg_lc():
    def __init__(self):
        self.model = YOLO('C:/Users/Admin/Downloads/best.pt')

    def find_lc(self,img):
        results = self.model.predict(source=img)
        ok=True
        if results:
            box = results[0].boxes
            if len(box.xyxy)!=0:
                box_xyxy = np.array(box.xyxy[0],dtype = int)
                box_xyxyn = np.array(box.xyxyn[0],dtype = float)
                return ok,box_xyxy,box_xyxyn
            else:
                return False,None,None
        else:
            return False,None,None



