from pydoc import plain
import sys
import cv2
import imageio
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





#with FaceLandmarksDetector("../test/close_eye.gif") as det:
    
det1 = FaceLandmarksDetector("../test/smile-wink1.mp4")
det2 = FaceLandmarksDetector("../test/smile-wink2.mp4")

i=0
zeman = cv2.imread("../test/zeman3.jpg")
#zeman = cv2.cvtColor(zeman, cv2.COLOR_BGR2RGBA)
zeman_face_landmarks = det1.process(zeman)
previous_zeman_face_landmarks = None
previous_face_landmarks1 = None
previous_face_landmarks2 = None


# create background image
zeman_bck = zeman.copy()
teeth_right = cv2.imread("teeth.jpg")
scale_factor = (zeman_face_landmarks[287][0] - zeman_face_landmarks[14][0]) / teeth_right.shape[1]
size = int(teeth_right.shape[1] * scale_factor), int(teeth_right.shape[0] * scale_factor)
teeth_right = cv2.resize(teeth_right, (size))

y_offset = int(zeman_face_landmarks[14][1] - teeth_right.shape[0]/2)
x_offset = int(zeman_face_landmarks[14][0])
zeman_bck[y_offset:y_offset+teeth_right.shape[0], x_offset:x_offset+teeth_right.shape[1]] = teeth_right

cv2.imwrite("teeth_test.jpg", zeman_bck)
#exit()


zeman = zeman // 2 + 127

pt1 = zeman_face_landmarks.to_numpy()[[33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]][:,0:2].astype(np.int32)
pt2 = zeman_face_landmarks.to_numpy()[[362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]][:,0:2].astype(np.int32)
pt3 = zeman_face_landmarks.to_numpy()[[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]][:,0:2].astype(np.int32)
cv2.fillPoly(zeman, pts=[pt1,pt2,pt3], color=(0, 0, 0))   # fill with transparency

#for (x,y,z) in zeman_face_landmarks.to_numpy():    
#    cv2.circle(zeman, (int(x),int(y)), 1, (255, 255, 255), -1)
#cv2.imwrite("test.png", zeman)
#xit()

#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('../test/output.avi', fourcc, 12, (zeman.shape[1],zeman.shape[0]))
#out = cv2.VideoWriter('../test/output.avi', fourcc, 4.0, (335,215))

result_list = []
vector_result = []
frame_rate = 8

i=0

for (_,face_landmarks1),(_,face_landmarks2) in zip(det1,det2): #[::int(24/frame_rate)]:
#for _,face_landmarks2 in det2:

    if previous_face_landmarks2 is None:
        previous_face_landmarks1 = face_landmarks1.copy()
        previous_face_landmarks2 = face_landmarks2.copy()
        previous_zeman_face_landmarks = zeman_face_landmarks.copy()
        continue
    
    vectors1 = face_landmarks1.get_normalized() - previous_face_landmarks1.get_normalized()
    vectors2 = face_landmarks2.get_normalized() - previous_face_landmarks2.get_normalized()
    
    vectors = vectors1 * 1.2 + vectors2 * 0.8
    
    vector_result.append(vectors)
    
    
    #vectors = vectors2 #todo
    
    
    #previous_face_landmarks = face_landmarks

    #TODO error shoulk translate previous zeman (1 frame landmarks) works only because it saves normalized landmarks fromn 1. image
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


    zem = cv2.remap(zeman, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_CONSTANT) 
 
    
    #result = zem.copy()
    #result[zem == 255] = zem*2
    #result = zem * 2 

    #print(zem.min(), zem.max(), np.count_nonzero(zem==1))
    #ker = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    #mask = cv2.erode(zem, ker) == 0
    #cv2.imwrite("../test/erode/mask" + str(i) + ".jpg", 255 * mask.astype(np.uint8))
    #print(i)

    result = np.where(zem>128, (zem-127)*2, zeman_bck)
    
    result_list.append(result)
    #out.write(result)
        
    #if i == 3: break
    i+=1
    
    #cv2.imwrite("test.png", img)
    #break
        
imageio.mimsave('../test/out.gif', result_list, fps=frame_rate)
#cap.release()
#cv2.destroyAllWindows()

vector_result = np.array(vector_result)

np.save("data.npy", vector_result)