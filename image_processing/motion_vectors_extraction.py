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

import os
import sys
import numpy as np
from face_utils.landmarks_detection import FaceLandmarksVideoDetector

def main():
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

if __name__ == "__main__":
    main()