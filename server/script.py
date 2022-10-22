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

#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_face_mesh = mp.solutions.face_mesh
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



with FaceLandmarksDetector() as det:
    i=0
    zeman = cv2.imread("../zeman4.jpg")
    #zeman = cv2.cvtColor(zeman, cv2.COLOR_BGR2RGBA)
    zeman_face_landmarks = det.process(zeman)

    previous_zeman_face_landmarks = None
    previous_face_landmarks = None

    zeman_bck = zeman.copy()
    zeman = zeman // 2
    pt1 = zeman_face_landmarks.to_numpy()[[33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]][:,0:2].astype(np.int32)
    pt2 = zeman_face_landmarks.to_numpy()[[362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]][:,0:2].astype(np.int32)
    pt3 = zeman_face_landmarks.to_numpy()[[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]][:,0:2].astype(np.int32)
    cv2.fillPoly(zeman, pts=[pt1,pt2,pt3], color=(255, 255, 255))   # fill with transparency
    #for (x,y,z) in zeman_face_landmarks.to_numpy():    
    #    cv2.circle(zeman, (int(x),int(y)), 1, (255, 255, 255), -1)
    #cv2.imwrite("test.png", zeman)
    #xit()

    cap = cv2.VideoCapture('../wink-smile.gif')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 8.0, (zeman.shape[1],zeman.shape[0]))

    while (cap.isOpened()):
        ret,frame = cap.read()
        if not ret: break
        #cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 3)

        face_landmarks = det.process(frame)
        face_landmarks.normalize()
        
        if previous_face_landmarks is None:
            previous_face_landmarks = face_landmarks
            previous_zeman_face_landmarks = zeman_face_landmarks.copy()
            continue
        
        vectors = face_landmarks.to_numpy() - previous_face_landmarks.to_numpy()
        previous_face_landmarks = face_landmarks
        zeman_face_landmarks.translate(vectors)

        #img = np.zeros(zeman.shape, dtype=np.uint8)
        #for (x,y,z) in zeman_face_landmarks.to_numpy():    
        #    #cv2.circle(img, (int(x),int(-z+700)), 1, (255, 255, 255), -1)
        #    cv2.circle(img, (int(x),int(y)), 1, (255, 255, 255), -1)

        # cut holes
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        #p1 = FaceLandmarksList(face_landmarks, img.shape).get_points([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7])
        #p2 = FaceLandmarksList(face_landmarks, img.shape).get_points([])
        #cv2.fillPoly(img, pts=[points], color=(255, 255, 255, 0))   # fill with transparency

        #x,y = np.meshgrid(np.arange(zeman.shape[1]), np.arange(zeman.shape[0]))
        #x,y = x.astype(np.float32), y.astype(np.float32)

        l = zeman_face_landmarks.to_numpy()
        pl = previous_zeman_face_landmarks.to_numpy()

        #vect = pl - l
        #print(vect.min(), vect.max())
        
        #previous_zeman_face_landmarks = zeman_face_landmarks.copy()

        #spline_x = interpolate.bisplrep(l[:,1], l[:,0], vect[:,0], s=30, kx=1, ky=1)
        #spline_y = interpolate.bisplrep(l[:,1], l[:,0], vect[:,1], s=30, kx=1, ky=1)
        #vect_x = interpolate.bisplev(np.arange(zeman.shape[0]), np.arange(zeman.shape[1]), spline_x).astype(np.float32)
        #vect_y = interpolate.bisplev(np.arange(zeman.shape[0]), np.arange(zeman.shape[1]), spline_y).astype(np.float32)
        
        #print(l[:,0:2].shape)
        #print(vect[:,0].shape)
        
        f_x = interpolate.LinearNDInterpolator(l[:,0:2], pl[:,0])
        x,y = np.meshgrid(np.arange(zeman.shape[1]), np.arange(zeman.shape[0]))
        vect_x = f_x(x,y)
        
        f_y = interpolate.LinearNDInterpolator(l[:,0:2], pl[:,1])
        x,y = np.meshgrid(np.arange(zeman.shape[1]), np.arange(zeman.shape[0]))
        vect_y = f_y(x,y)

        #x,y = np.meshgrid(np.arange(zeman.shape[1]), np.arange(zeman.shape[0]))
        #x,y = x.astype(np.float32), y.astype(np.float32)
        x = vect_x.astype(np.float32)
        y = vect_y.astype(np.float32)

        #cv2.imwrite('x.png', vect_x.astype(np.uint8))
        #cv2.imwrite('y.png', vect_y.astype(np.uint8))
        #break

        #print(x.min(), x.max(), y.min(), y.max())
        #break

        #x,y = np.meshgrid(np.arange(zeman.shape[1]), np.arange(zeman.shape[0]))
        #x,y = x.astype(np.float32), y.astype(np.float32)
        
        zem = cv2.remap(zeman, x, y, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT) 

        #result = zeman.copy()
        #result[zem < 128] = zem*2

        #result = zem * 2 
        result = np.where(zem<128, zem*2, zeman_bck)

        #for (x,y,z) in l:    
        #    cv2.circle(zeman, (int(x),int(y)), 1, (255, 255, 255), -1)

        #cv2.imwrite("test3/" + str(i) + ".png", img)
        #zem = cv2.cvtColor(zem, cv2.COLOR_RGBA2BGR)
        out.write(result)
        
        #if i == 3: break
        i+=1
        
        #cv2.imwrite("test.png", img)
        #break
        
        
cap.release()
cv2.destroyAllWindows()