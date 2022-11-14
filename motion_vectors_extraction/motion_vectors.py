from os
import sys
import numpy as np
from ../image_processing/face_landmarks import FaceLandmarksVideoDetector

# process input arguments 
# --dst dst_path --videos video_path_1 [weight_1] [video_path_2 [weight_2] ...]
dst_path = sys.argv[sys.argv.index("--dst_path") + 1]
detectors = []
argv = sys.argv[(sys.argv.index("--videos")+1):]
for i, arg in enumerate(argv):
    if os.path.exists(arg):
        detector = FaceLandmarksVideoDetector(arg)
        try: weight = float(argv[i+1])
        except: weight = 1
        detectors.append((detector, weight))

initial_face_landmarks = [ det.next() for det in detectors ]

vectors_result = []
while 1:
    vectors = [ w * (det.next().get_normalized() - ifl.get_normalized()) for ifl,(det,w) in zip(initial_face_landmarks,detectors) ]
    vectors_result.append(sum(vectors))

np.save(dst_path, np.array(vectors_result))