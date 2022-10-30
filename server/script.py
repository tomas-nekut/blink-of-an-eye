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


zeman = cv2.imread("../test/zeman5.jpg")
zeman = cv2.cvtColor(zeman, cv2.COLOR_BGR2RGB)
zeman_face_landmarks = FaceLandmarksDetector().process(zeman)
pl = zeman_face_landmarks.to_numpy()

# create background image
zeman_bck = zeman.copy()
teeth_right = cv2.imread("teeth.jpg")
scale_factor = (zeman_face_landmarks[287][0] - zeman_face_landmarks[14][0]) / teeth_right.shape[1]
size = int(teeth_right.shape[1] * scale_factor), int(teeth_right.shape[0] * scale_factor)
teeth_right = cv2.resize(teeth_right, (size))
y_offset = int(zeman_face_landmarks[14][1] - teeth_right.shape[0]/2)
x_offset = int(zeman_face_landmarks[14][0])
zeman_bck[y_offset:y_offset+teeth_right.shape[0], x_offset:x_offset+teeth_right.shape[1]] = teeth_right

pt1 = zeman_face_landmarks.to_numpy()[[33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]][:,0:2].astype(np.int32)
pt2 = zeman_face_landmarks.to_numpy()[[362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]][:,0:2].astype(np.int32)
pt3 = zeman_face_landmarks.to_numpy()[[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]][:,0:2].astype(np.int32)
cv2.fillPoly(zeman, pts=[pt1,pt2,pt3], color=(0, 0, 0))   # fill with transparency

result_list = []
frame_rate = 8

for vectors in tqdm(np.load("data.npy")[::int(24/frame_rate)]):

    l = zeman_face_landmarks.copy().translate(vectors).to_numpy()
   
    f_x = interpolate.LinearNDInterpolator(l[:,0:2], pl[:,0])
    x,y = np.meshgrid(np.arange(zeman.shape[1]), np.arange(zeman.shape[0]))
    vect_x = f_x(x,y)
    
    f_y = interpolate.LinearNDInterpolator(l[:,0:2], pl[:,1])
    x,y = np.meshgrid(np.arange(zeman.shape[1]), np.arange(zeman.shape[0]))
    vect_y = f_y(x,y)
   
    x = vect_x.astype(np.float32)
    y = vect_y.astype(np.float32)
    
    

    remaped_zeman = cv2.remap(zeman, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_CONSTANT) 
    #remap_region =  np.dstack([np.logical_not(np.isnan(x) | np.isnan(y))]*3)
    
    
    #for (x,y,z) in l:    
    #   cv2.circle(result, (int(x),int(y)), 1, (255, 255, 255), -1)


    #holes = np.zeros_like(zeman)
    pt1 = l[[33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]][:,0:2].astype(np.int32)
    pt2 = l[[362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]][:,0:2].astype(np.int32)
    pt3 = l[[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]][:,0:2].astype(np.int32)
    cv2.fillPoly(remaped_zeman, pts=[pt1,pt2,pt3], color=(0, 0, 0))  
    #holes = holes.astype(bool)

    #print(remap_region.dtype, holes.dtype)

    #mask = remap_region & np.logical_not(holes)
    
    result = np.where(remaped_zeman!=0, remaped_zeman, zeman_bck)


    '''
    for (x,y,z) in face_landmarks2.get_normalized():    
        cv2.circle(result, (int(x*500)+250,int(y*500)+250), 1, (255, 255, 255), -1)
 
    for (x,y,z) in previous_face_landmarks2.get_normalized():    
        cv2.circle(result, (int(x*500)+250,int(y*500)+250), 1, (255, 0, 0), -1)
    '''

    
    result_list.append(result)
       
imageio.mimsave('../test/out.gif', result_list, fps=frame_rate)
