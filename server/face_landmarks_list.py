# https://github.com/nghiaho12/rigid_transform_3D
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html


import numpy as np
#from skimage import transform as tf
import mediapipe as mp
#from rigid_transform_3D import rigid_transform_3D
#from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R


class FaceLandmarksDetector(mp.solutions.face_mesh.FaceMesh):
    def __init__(self):
        super().__init__(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    def process(self, img):
        landmarks = super().process(img).multi_face_landmarks[0].landmark[:468]
        return FaceLandmarksList(landmarks, img.shape)
    
class FaceLandmarksList():
    def __init__(self, landmarks, img_shape):
        if type(landmarks) is np.array:
            self.__landmarks = landmarks
        else:
            self.__landmarks = np.array([(l.x*img_shape[1], l.y*img_shape[0], l.z*img_shape[1]) for l in landmarks])
        self.__img_shape = img_shape
        #self.__img_shape = img_shape
    def __getitem__(self, index):
        #if type(index) is tuple:
        #    x = 0.5 * (self.__landmarks[index[0]].x + self.__landmarks[index[1]].x)
        #    y = 0.5 * (self.__landmarks[index[0]].y + self.__landmarks[index[1]].y)
        #else:
        #    x = self.__landmarks[index].x 
        #    y = self.__landmarks[index].y
        #x = int(self.__landmarks[index][0] * self.__img_shape[1])
        #y = int(self.__landmarks[index][1] * self.__img_shape[0])
        #z = int(self.__landmarks[index][2] * self.__img_shape[1])
        return self.__landmarks[index]
    
    def copy(self):
        l = FaceLandmarksList([], self.__img_shape)
        l.from_numpy(self.__landmarks)
        return l

    def translate(self, vectors):
        self.normalize()
        self.__landmarks += vectors
        self.denormalize()

    def normalize(self):
        # rotation
        angle_x = -self.__angle((self[200]-self[9])[[1,2]],   np.array([1,0]))
        angle_y = -self.__angle((self[447]-self[227])[[0,2]], np.array([1,0]))
        angle_z = -self.__angle((self[447]-self[227])[[0,1]], np.array([1,0]))
        self.__norm_rotation = R.from_euler('xyz', [angle_x,angle_y,angle_z])
        self.__landmarks = self.__norm_rotation.apply(self.__landmarks)
        # scale
        self.__norm_scale = 1/(self.__landmarks.max(axis=0)-self.__landmarks.min(axis=0))
        self.__landmarks *= self.__norm_scale
        # translation
        self.__norm_translate = -self[4]
        self.__landmarks += self.__norm_translate
        
    def denormalize(self):
        self.__landmarks -= self.__norm_translate
        self.__landmarks /= self.__norm_scale
        self.__landmarks = self.__norm_rotation.apply(self.__landmarks, inverse=True)
        
    def __angle(self, a, b):
        return np.arccos(np.dot(a,b) / np.abs(np.linalg.norm(a) * np.linalg.norm(b)))
    
    def to_numpy(self):
        return np.array(list(self))

    def from_numpy(self, landmarks):
        self.__landmarks = landmarks
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


