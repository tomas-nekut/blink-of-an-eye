# In the wink of an eye
# an extension for Google Chrome that makes images of 
# Czech president Miloš Zeman more realistic
#
# Copyright (C) 2022 Tomáš Nekut
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You received a copy of the GNU General Public License
# along with this program or you can find it at <https://www.gnu.org/licenses/>.

import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
from forbiddenfruit import curse

curse(np.ndarray, 'to_XY', lambda x: x[:,0:2])

class FaceLandmarksDetector(mp.solutions.face_mesh.FaceMesh):
    def __init__(self):
        super().__init__(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    def __process_numpy(self, img):
        landmarks = super().process(img).multi_face_landmarks[0].landmark
        landmarks = np.array([(l.x*img.shape[1], l.y*img.shape[0], l.z*img.shape[1]) for l in landmarks])
        return landmarks
    
    def process(self, img):
        landmarks = self.__process_numpy(img)
        return FaceLandmarksList(landmarks)

class FaceLandmarksVideoDetector(FaceLandmarksDetector):
    def __init__(self, video_path=None, smoothing_cnt=5):
        super().__init__()
        if video_path != None:
            self.open(video_path)
        self.__smoothing_cnt = smoothing_cnt
    
    def open(self, video_path):
        self.__cap = cv2.VideoCapture(video_path)
    
    def get_frame_width(self):
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_size(self):
        return self.get_frame_width(), self.get_frame_height()

    def __iter__(self):
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.__landmarks = None
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if not self.__cap.isOpened():
            raise StopIteration()
        ret,frame = self.__cap.read()
        if not ret:
            raise StopIteration()
        landmarks = self.__process_numpy(frame)
        landmarks = self.__smooth(landmarks)
        return FaceLandmarksList(landmarks)

    def __smooth(self, landmarks):
        self.__push_back(landmarks)
        return self.__landmarks.mean(axis=0)

    def __push_back(self, landmarks):
        landmarks = np.expand_dims(landmarks, axis=0)
        if self.__landmarks is None:
            self.__landmarks = landmarks
        else:
            self.__landmarks = np.concatenate([self.__landmarks, landmarks])  
            if self.__landmarks.shape[0] > self.__smoothing_cnt:
                self.__landmarks = self.__landmarks[1:]
            
class FaceLandmarksList():
    def __init__(self, landmarks):
        self.__landmarks = landmarks[:468]  # only mesh
        self.__normalized_landmarks = None

    def __getitem__(self, index):
        return self.__landmarks[index]

    def copy(self):
        return FaceLandmarksList(self.__landmarks)

    def get_mesh(self):
        return self.__landmarks

    def get_left_eye_outline(self):
        return self.__landmarks[[33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]]

    def get_right_eye_outline(self):
        return self.__landmarks[[362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]]

    def get_mouth_outline(self):
        return self.__landmarks[[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]]
   
    def get_teeth_line(self, type='middle'):
        teeth_line_left = [[-0.23, 0.398, 0.45], [-0.22, 0.396, 0.32], [-0.18, 0.394, 0.18], [-0.1, 0.392, 0.1], [0., 0.39, 0.05]]
        teeth_line_right = [(-x,y,z) for (x,y,z) in teeth_line_left[3::-1]]
        teeth_line = teeth_line_left + teeth_line_right
        teeth_line = np.array(teeth_line)
        if type=='upper':
            teeth_line = teeth_line + np.array([0,-0.06,0])
        elif type=='lower':
            teeth_line = teeth_line + np.array([0, 0.06,0])
        return self.__denormalize(teeth_line)

    def get_mouth_openness_coef(self):
        return np.linalg.norm((self[14]-self[13])) / np.linalg.norm(self[308]-self[78])

    def get_right_eye_openness_coef(self):
        return np.linalg.norm((self[386]-self[374])) / np.linalg.norm(self[263]-self[362])

    def translate(self, vectors, exclude_boudary=True):
        # prevent boundary landmarks from moving by clearing their translation vectors
        if exclude_boudary:
            vectors[[127,162,21,54,103,67,109,10,338,297,332,284,251,389,368,264,356,447,454,366,323,401,361,435,288,367,397,364,365,394,379,378,400,377,152,148,176,149,150,169,136,135,172,138,58,215,177,132,137,93,227,234,34]] = 0
        # apply translation to normalized landmarks
        landmarks = self.get_normalized()
        landmarks += vectors
        # move certain landmarks that tends to overlap and cause interpolation problem 
        for u,l in [(398,381),(384,380),(385,380),(385,374),(386,374),(386,373),(387,373),(387,390),(388,249)]:
            if landmarks[u][1] + 0.01 > landmarks[l][1]:
                landmarks[u][1] = landmarks[l][1] - 0.01
        for u,l in [(17,18),(84,83),(181,182),(314,313),(405,406)]:
            if landmarks[u][1] + 0.02 > landmarks[l][1]:
                landmarks[l][1] = landmarks[u][1] + 0.02
        # denormalize
        landmarks = self.__denormalize(landmarks)
        return FaceLandmarksList(landmarks)

    def get_normalized(self):
        normalized_landmarks, _ = self.__compute_normalization()
        return normalized_landmarks

    def __compute_normalization(self):
        if self.__normalized_landmarks is None:
            # rotation
            vec1 = self[447]-self[227]
            vec2 = self[200]-self[9]
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)
            a = np.array((vec1,vec2,np.cross(vec1,vec2)))
            b = np.array(((1,0,0),(0,1,0),(0,0,1)))
            h = b @ a.T
            u,s,v = np.linalg.svd(h)
            r = v.T @ u.T
            self.__norm_rotation = R.from_matrix(r)
            self.__normalized_landmarks = self.__norm_rotation.apply(self.__landmarks)
            # scale
            self.__norm_scale = 1/(self.__normalized_landmarks.max(axis=0)-self.__normalized_landmarks.min(axis=0))
            self.__normalized_landmarks *= self.__norm_scale
            # translation
            self.__norm_translate = -self.__normalized_landmarks[6] 
            self.__normalized_landmarks += self.__norm_translate
        return self.__normalized_landmarks.copy(), (self.__norm_translate, self.__norm_scale, self.__norm_rotation)
        
    def __denormalize(self, points):
        _, (norm_translate, norm_scale, norm_rotation) = self.__compute_normalization()
        points = points.copy()
        points -= norm_translate
        points /= norm_scale
        points = norm_rotation.apply(points, inverse=True)
        return points
        
