from itertools import tee
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
from tqdm import tqdm

img = cv2.imread("../test/zeman3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
landmarks = FaceLandmarksDetector().process(img)


def add_teeth(original_img):
    img = original_img.copy()
    teeth = cv2.imread("teeth.jpg")
    teeth = cv2.cvtColor(teeth, cv2.COLOR_BGR2RGB)

    bg = img

    t = np.zeros_like(img)
    t[0:teeth.shape[0], 0:teeth.shape[1]] = teeth


    h = FaceLandmarksDetector().process(bg).get_teeth_line()
    # find local max
    min = h[:,0].argmin()
    max = h[:,0].argmax()
    teeth_line_upper  = landmarks.get_teeth_line('upper') [min:max+1]
    teeth_line_middle = landmarks.get_teeth_line('middle')[min:max+1]
    teeth_line_lower  = landmarks.get_teeth_line('lower') [min:max+1]
    teeth_line = np.vstack([teeth_line_upper,teeth_line_middle,teeth_line_lower])
    h = teeth_line[:,:2]

    root = lambda x: np.sign(x) * (abs(x)**(0.7))
    c = np.linspace(0,1,9)  # len landmarks.get_teeth_line()
    c = ((root(2*c-1)))*(1/2)+(1/2)
    print(c)
    c *= teeth.shape[1]
    c = c[min:(max+1)]
    print(c)
    print("teeth.shape[1]", teeth.shape[1])
    g = np.expand_dims(c, axis=1)

    g = np.hstack([g, np.ones_like(g)*teeth.shape[0]/2]).astype(np.int32)
    a = g.copy()
    a[:,1] = 0
    b = g.copy()
    b[:,1] = teeth.shape[0]
    g = np.vstack([a,g,b])

    print(h.shape)
    print(g.shape)



    #for v,b in zip(h,g):
    #    cv2.line(bg, v.astype(np.int32), b.astype(np.int32), (255,255,255), 1)
        

    f_x = interpolate.LinearNDInterpolator(h, g[:,0])
    f_y = interpolate.LinearNDInterpolator(h, g[:,1])
    x = f_x(*np.meshgrid(np.arange(t.shape[1]), np.arange(t.shape[0]))).astype(np.float32)
    y = f_y(*np.meshgrid(np.arange(t.shape[1]), np.arange(t.shape[0]))).astype(np.float32)
    remaped_t = cv2.remap(t, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_CONSTANT) 

    bg = np.where(remaped_t != 0, remaped_t, bg)
    return bg

'''
for (x,y) in h:    
    cv2.circle(remaped_t, (int(x),int(y)), 1, (255, 255, 255), -1)

for (x,y) in g:    
    cv2.circle(remaped_t, (int(x),int(y)), 1, (255, 255, 255), -1)
'''

#cv2.imwrite("../test/bg.jpg", bg)
#exit()

bg = add_teeth(img)

# fill eyes and mouth with black
poly = [landmarks.get_left_eye_outline(), landmarks.get_right_eye_outline(), landmarks.get_mouth_outline()]
cv2.fillPoly(img, pts=poly, color=(0, 0, 0))  

result_list = []
frame_rate = 8

for vectors in tqdm(np.load("motion_vectors.npy")[::int(24/frame_rate)]):

    translated_landmarks = landmarks.translate(vectors)
   
    # interpolate 
    f_x = interpolate.LinearNDInterpolator(translated_landmarks.to_numpy()[:,0:2], landmarks.to_numpy()[:,0])
    f_y = interpolate.LinearNDInterpolator(translated_landmarks.to_numpy()[:,0:2], landmarks.to_numpy()[:,1])
    
    # evaluete interpolation function for each pixel of input image
    x = f_x(*np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))).astype(np.float32)
    y = f_y(*np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))).astype(np.float32)
   
    remaped_img = cv2.remap(img, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_CONSTANT) 

    poly = [translated_landmarks.get_left_eye_outline(), translated_landmarks.get_right_eye_outline(), translated_landmarks.get_mouth_outline()]
    cv2.fillPoly(remaped_img, pts=poly, color=(0, 0, 0))  
   
    result = np.where(remaped_img != 0, remaped_img, bg)
    result_list.append(result)
       
imageio.mimsave('../test/out.gif', result_list, fps=frame_rate)
