import sys
#import torch
import cv2
from scipy import interpolate
import numpy as np
from flask import Flask, Response, request
import requests
import io
#import dlib
#from imutils import face_utils
import face_recognition
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mediapipe as mp
from face_landmarks_list import FaceLandmarksList
 
port = 50000
app = Flask(__name__)
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("./dlib/shape_predictor_68_face_landmarks.dat")

#@app.route('/', methods=['POST'])
#img = cv2.imdecode(np.fromstring(request.data, np.uint8), cv2.IMREAD_COLOR)

#http://localhost:50000/https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2F0%2F00%2FMilo%25C5%25A1_Zeman_2022.jpg%2F225px-Milo%25C5%25A1_Zeman_2022.jpg
#localhost:50000/https%3A%2F%2Fwww.irozhlas.cz%2Fsites%2Fdefault%2Ffiles%2Fstyles%2Fzpravy_rubrikovy_nahled_vyskovy%2Fpublic%2Fuploader%2F2017-09-19t160326z_9_171010-173745_miz.jpg%3Fitok%3DGiGaCzML
#http://localhost:50000/https%3A%2F%2F1gr.cz%2Ffotky%2Fidnes%2F22%2F093%2Fcl8h%2FEPC964a21_pou_zeman.jpg



map = np.zeros((30,30)) + np.linspace(0, 10, 30).reshape((30,1))


def visualize_warped_mesh(mesh):
    spacing = 5
    img = np.ones(mesh.shape[0:2], dtype=np.uint8) * 255
    # Draw grid
    img[:, spacing:-1:spacing] = 0
    img[spacing:-1:spacing, :] = 0   
    img = cv2.remap(img, mesh, None, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT) 
    return img




left_lower_eyelid = [(33,33),(246,246),(161,161),(160,160),(159,159),(158,158),(157,157),(173,173),(133,133),((7,246),7),((163,161),163),((144,160),144),((145,159),145),((153,158),153),((154,157),154),((155,173),155),(243,243),(128,128),(47,47),(100,100),(118,118),(31,31),(130,130), (23,(23,230)), (110,(110,229)), (26,(26,232))]
#left_lower_eyelid = [(33,33),(246,246),(161,161),(160,160),(159,159),(158,158),(157,157),(173,173),(133,133),((145,159),145),(243,243),(128,128),(47,47),(100,100),(118,118),(31,31),(130,130)]
#left_lower_eyelid = [(33,33),(246,246),(161,161),(160,160),(159,159),(158,158),(157,157),(173,173),(133,133),((144,160),144),((153,158),153),(243,243),(128,128),(47,47),(100,100),(118,118),(31,31),(130,130)]

#52-105
left_upper_eyelid = [(33,33),(130,130),(113,113),(124,124),(71,71),(68,68),(104,104),(69,69),(108,108),(9,9),(8,8),(168,168),(122,122),(245,245),(244,244),(243,243),(133,133),(155,155),(154,154),(153,153),(145,145),(144,144),(163,163),(7,7),((155,173),173),((154,157),157),((153,158),158),((145,159),159),((144,160),160),((163,161),161),((7,246),246),(46,70),(53,63),(52,105),(65,66)]

@app.route('/<path:url>', methods=['GET'])
def get_bolts_bitmap_generator_version(url):

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)

    

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        face_landmarks = face_mesh.process(img).multi_face_landmarks[0]
        
        #for (x,y) in FaceLandmarksList(face_landmarks, img.shape):    
        #    cv2.circle(img, (x,y), 1, (255, 255, 255), -1)

        # cut holes
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        points = FaceLandmarksList(face_landmarks, img.shape).get_points([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7])
        cv2.fillPoly(img, pts=[points], color=(255, 255, 255, 0))   # fill with transparency
        
        
        x,y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        x,y = x.astype(np.float32), y.astype(np.float32)

        #l = FaceLandmarksList(face_landmarks, img.shape)
        ## ~119->230
        #p = l.get_polygon([230,133,243,128,47,100,118,31,130,33,7,163,144,145,153,154,155])
        #p_x = p[:,0]
        #p_y = p[:,1]
        #p_z = [0]*len(p)
        #p_z[0] = 10
        #f = interpolate.interp2d(p_x, p_y, p_z)
        #vects_x = f(np.arange(img.shape[1]), np.arange(img.shape[0]))
        #p_z[0] = 5
        #f = interpolate.interp2d(p_x, p_y, p_z)
        #vects_y = f(np.arange(img.shape[1]), np.arange(img.shape[0]))

        
        landmarks = FaceLandmarksList(face_landmarks, img.shape)
        #vects = landmarks.get_vectors(left_lower_eyelid)
        vects = landmarks.get_vectors(left_upper_eyelid)

        print(vects)

        print('x inderp ', vects[:,0], vects[:,1], vects[:,2])

        spline_x = interpolate.bisplrep(vects[:,1], vects[:,0], vects[:,2], s=30, kx=2, ky=2)
        spline_y = interpolate.bisplrep(vects[:,1], vects[:,0], vects[:,3], s=30, kx=2, ky=2)
        vects_x = interpolate.bisplev(np.arange(img.shape[0]), np.arange(img.shape[1]), spline_x)
        vects_y = interpolate.bisplev(np.arange(img.shape[0]), np.arange(img.shape[1]), spline_y)

        print('vects_x.shape', vects_x.shape)

        
        
        #visualize
        vv_x = vects_x.copy()*5 + 127
        vv_y = vects_y.copy()*5 + 127
        #v = vects_x.astype(np.uint8) #.copy()
        #v = np.stack([v,v,v], axis=-1)
        #cv2.polylines(v, pts=[p], isClosed=True, color=(255,255,255))   
        cv2.imwrite('vects_x.png', vv_x.astype(np.uint8))
        cv2.imwrite('vects_y.png', vv_y.astype(np.uint8))
        i = vv_x.copy() #255*np.ones(img.shape)
        for v in vects:
            cv2.line(i, tuple(v[:2]), tuple(v[:2]+v[2:]), 0, 1) 
        cv2.imwrite('vects.png', i.astype(np.uint8))
        
        print('interpolated x', np.min(vects_x), '-', np.max(vects_x))
        print('interpolated y', np.min(vects_y), '-', np.max(vects_y))
       
        x += vects_x
        y += vects_y
        img = cv2.remap(img, x, y, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT) 
        #cv2.polylines(img, pts=[p[1:]], isClosed=True, color=(255,255,255,255))   

        for v in vects:
            cv2.line(img, tuple(v[:2]), tuple(v[:2]+v[2:]), 0, 1) 

        #img = visualize_warped_mesh(mesh)

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = bytearray(cv2.imencode('.png', img)[1])
    return Response(img, content_type="image/png")
    
if __name__ == '__main__':
    app.run(port=port, debug=True)






# blink detection  https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# python cloud https://dev.to/yash_makan/4-best-python-web-app-hosting-services-for-freewith-complete-process-57nb
# dlib tutorial http://dlib.net/face_landmark_detection.py.html
# face-recognition landm,ars, identification tuto https://pypi.org/project/face-recognition/


'''
{'chin': [(347, 378), (361, 414), (381, 449), (403, 482), (427, 511), (454, 535), (483, 553), (516, 563), (551, 558), (586, 542), (618, 517), (648, 487), (669, 449), (675, 404), (667, 355), (653, 306), (640, 258)], 'left_eyebrow': [(345, 341), (352, 316), (373, 302), (398, 297), (425, 294)], 'right_eyebrow': [(459, 272), (484, 250), (513, 234), (547, 231), (575, 245)], 'nose_bridge': [(451, 304), (458, 332), (465, 361), (472, 390)], 'nose_tip': [(456, 420), (473, 420), (491, 418), (506, 407), (520, 396)], 'left_eye': [(381, 346), (392, 327), (411, 318), (431, 327), (417, 340), (399, 347)], 'right_eye': [(499, 298), (511, 276), (530, 268), (549, 273), (539, 287), (519, 295)], 'top_lip': [(461, 483), (474, 470), (488, 460), (504, 458), (519, 447), (545, 441), (577, 436), (571, 440), (525, 456), (509, 465), (494, 468), (468, 481)], 'bottom_lip': [(577, 436), (558, 457), (539, 470), (523, 478), (507, 483), (487, 487), (461, 483), (468, 481), (499, 468), (514, 464), (530, 457), (571, 440)]}
chin
left_eyebrow
right_eyebrow
nose_bridge
nose_tip
left_eye
right_eye
top_lip
bottom_lip
'''


'''
    faces = detector(img)
    for face in faces:
        landmarks = predictor(img, face)
        landmarks = face_utils.shape_to_np(landmarks)
		for (x, y) in landmarks:
			cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
'''

'''
    faces = face_recognition.face_landmarks(img)
    for face in faces:

        right_eye_landmarks = []
        for n,l in face.items():
            print(n)
            right_eye_landmarks.extend(l)


        #right_eye_landmarks = face['left_eye']


        #right_eye_landmark = right_eye_landmarks[0]
        right_eye_landmark = face['left_eye'][0]
'''

'''

x,y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        x,y = x.astype(np.float32), y.astype(np.float32)
        
        
        #mesh[right_eye_landmark[1]:right_eye_landmark[1]+30, right_eye_landmark[0]:right_eye_landmark[0]+30] = mesh[right_eye_landmark[1]+30:right_eye_landmark[1]+60, right_eye_landmark[0]+30:right_eye_landmark[0]+60]

        y[right_eye_landmark[1]:right_eye_landmark[1]+30, right_eye_landmark[0]:right_eye_landmark[0]+30] += map

        mesh = np.stack((x,y), axis=2).astype(np.float32)
        
        #mesh, None
        img = cv2.remap(img, x, y, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT) 
        

'''