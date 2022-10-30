# https://github.com/nghiaho12/rigid_transform_3D
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

# Oru Adaar Love, Manikya Malaraya Poovi Song Video
# expresions https://www.youtube.com/watch?v=embYkODkzcs MDI Management Development International

from math import degrees
from re import T
import numpy as np
#from skimage import transform as tf
import mediapipe as mp
#from rigid_transform_3D import rigid_transform_3D
#from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R
import cv2


class FaceLandmarksDetector(mp.solutions.face_mesh.FaceMesh):
    def __init__(self, video_path=None, smoothing_cnt=5):
        super().__init__(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        if video_path != None:
            self.open(video_path)
        self.__smoothing_cnt = smoothing_cnt
    def open(self, video_path):
        self.__cap = cv2.VideoCapture(video_path)
    def __iter__(self):
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.__landmarks = None
        #print("iter")
        return self
    def __next__(self):
        return self.next()
    def next(self):
        if not self.__cap.isOpened():
            raise StopIteration()
        ret,frame = self.__cap.read()
        if not ret:
            #print("stop")
            raise StopIteration()

        #print("frame stats", frame.min(), " ", frame.max(), " ", frame.mean() )

        landmarks = self.__process_numpy(frame)

        #print("landmarks:", landmarks.shape)

        landmarks = self.__smooth(landmarks)
        return frame, FaceLandmarksList(landmarks)

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
            
    def __process_numpy(self, img):
        landmarks = super().process(img).multi_face_landmarks[0].landmark
        landmarks = np.array([(l.x*img.shape[1], l.y*img.shape[0], l.z*img.shape[1]) for l in landmarks])
        return landmarks
    
    def process(self, img):
        landmarks = self.__process_numpy(img)
        return FaceLandmarksList(landmarks)
    
    def get_frame_width(self):
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        return int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_size(self):
        return self.get_frame_width(), self.get_frame_height()

    
class FaceLandmarksList():
    def __init__(self, landmarks):
        self.__landmarks = landmarks[:468]  # only mesh
        self.__normalized_landmarks = None
        #self.mesh = self.__landmarks[:468]
        #self.left_iris = self.__landmarks[468]
        #self.right_iris = self.__landmarks[469]

    def __getitem__(self, index):
        return self.__landmarks[index]
    
    def copy(self):
        return FaceLandmarksList(self.__landmarks)

    def translate(self, vectors, exclude_boudary=True):
        if exclude_boudary:
            vectors[[127,162,21,54,103,67,109,10,338,297,332,284,251,389,368,264,356,447,454,366,323,401,361,435,288,367,397,364,365,394,379,378,400,377,152,148,176,149,150,169,136,135,172,138,58,215,177,132,137,93,227,234,34]] = 0
        landmarks = self.get_normalized()
        landmarks += vectors

        for u,l in [(398,381),(384,380),(385,374),(386,373),(387,390),(388,249)]:
            if landmarks[u][1] + 0.01 > landmarks[l][1]:
                landmarks[u][1] = landmarks[l][1] - 0.01

        for u,l in [(17,18),(84,83),(181,182),(314,313),(405,406)]:
            if landmarks[u][1] + 0.02 > landmarks[l][1]:
                landmarks[l][1] = landmarks[u][1] + 0.02
                
        landmarks -= self.__norm_translate
        landmarks /= self.__norm_scale
        landmarks = self.__norm_rotation.apply(landmarks, inverse=True)
        self.__landmarks = landmarks
        return self

    def get_normalized(self):
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
            self.__norm_translate = -self.__normalized_landmarks[6] #4
            self.__normalized_landmarks += self.__norm_translate
        return self.__normalized_landmarks.copy()
        
    #def denormalize(self):
    #    self.__landmarks -= self.__norm_translate
    #    self.__landmarks /= self.__norm_scale
    #    self.__landmarks = self.__norm_rotation.apply(self.__landmarks, inverse=True)
        
    def __angle(self, a, b):
        return np.arccos(np.dot(a,b) / np.abs(np.linalg.norm(a) * np.linalg.norm(b)))
    
    def to_numpy(self):
        return np.array(list(self))

    #def from_numpy(self, landmarks):
    #    self.__landmarks = landmarks
    '''
    def get_points_generator(self, landmark_indexes):
        for i in landmark_indexes:
            yield self[i]
    def get_points(self, landmark_indexes):
        return np.array(list(self.get_points_generator(landmark_indexes)))
    def get_vectors(self, landmark_vector_indexes):
        # landmark_vector_indexes is [(landmark_indexe1, landmark_indexe2, norm_length), ...] 
        # where 
        vects = [ self[l1] + self[l2] for l1,l2 in landmark_vector_indexes ]
        vects = np.array(vects)
        a = np.zeros(vects.shape, dtype=np.int32)
        a[:,0] = vects[:,0]
        a[:,1] = vects[:,1]
        a[:,2] = vects[:,2] - vects[:,0]
        a[:,3] = vects[:,3] - vects[:,1]
        return a
    '''

    


    def __normalize(self):
        src = self.get_points([197,4])
        dst = np.array([[500,0],[500,200]])
        tform = tf.estimate_transform('similarity', src, dst)
       
        print(src)
        print(dst) 
        print(tform.params)

        mt = tf.matrix_transform(self.to_numpy(), tform.params)#mt is the same dst
        #mean_squared_error(mt,dst) #should be zero
        #print( '{:.10f}'.format(mean_squared_error(mt,dst)) )
        return mt


