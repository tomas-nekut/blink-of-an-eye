from pydoc import plain
import sys
import cv2
from scipy import interpolate
import numpy as np
from flask import Flask, Response, request
import requests
import io
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from face_landmarks_list import FaceLandmarksDetector


with FaceLandmarksDetector("../test/smile3.gif") as det:
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('../test/output_test.avi', fourcc, 8.0, det.get_frame_size())

    for frame,face_landmarks in det:
        
        for (x,y,z) in face_landmarks.to_numpy()[:469]:    
            cv2.circle(frame, (int(x),int(y)), 1, (255, 255, 255), -1)
        
        #print(face_landmarks.get_angle_z())

        #cv2.putText(frame, str(face_landmarks.get_angle_z()/np.pi*180)+"°", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        out.write(frame)

        

    