
import numpy as np
from face_landmarks import FaceLandmarksVideoDetector

det_1 = FaceLandmarksVideoDetector("./motion_source_videos/1.mp4")
det_2 = FaceLandmarksVideoDetector("./motion_source_videos/2.mp4")

vectors_result = []

for i,(face_landmarks_1,face_landmarks_2) in enumerate(zip(det_1,det_2)):
    if i==0:
        initial_face_landmarks_1 = face_landmarks_1.copy()
        initial_face_landmarks_2 = face_landmarks_2.copy()
        continue 
    vectors_1 = face_landmarks_1.get_normalized() - initial_face_landmarks_1.get_normalized()
    vectors_2 = face_landmarks_2.get_normalized() - initial_face_landmarks_2.get_normalized()    
    vectors = vectors_1 * 1.2 + vectors_2 * 0.8 
    vectors_result.append(vectors)

np.save("motion_vectors.npy", np.array(vectors_result))