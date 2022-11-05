import cv2
import imageio
from scipy import interpolate
import numpy as np
from face_landmarks import FaceLandmarksDetector
from tqdm import tqdm

# mapping from input to target image is described by set of source point -> destination point pairs 
def remap_image(img, src_points, dst_points):
    # interpolate source point for each pixel of the image
    f_x = interpolate.LinearNDInterpolator(dst_points, src_points[:,0])
    f_y = interpolate.LinearNDInterpolator(dst_points, src_points[:,1])
    x = f_x(*np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))).astype(np.float32)
    y = f_y(*np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))).astype(np.float32)
    # transform image
    remaped_img = cv2.remap(img, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_CONSTANT) 
    return remaped_img

# 
def add_teeth(original_img):
    teeth = cv2.imread("teeth.jpg")
    teeth = cv2.cvtColor(teeth, cv2.COLOR_BGR2RGB)
    teeth = cv2.resize(teeth, dsize=(original_img.shape[1],original_img.shape[0]))
    landmarks = FaceLandmarksDetector().process(original_img)
    # transform teeth image to match mouth position of face in original image
    # find part of teeth line that is visible given view angle
    teeth_line_start = landmarks.get_teeth_line()[:,0].argmin()
    teeth_line_end = landmarks.get_teeth_line()[:,0].argmax() + 1 
    # find source points of transformation 
    # src points are nelineary positioned to compensate for frontal teeth image (not panoramatic)
    src_points = np.linspace(0, 1, len(landmarks.get_teeth_line()))  
    root = lambda x: np.sign(x) * (abs(x)**(0.7))
    src_points = ((root(2*src_points-1)))*(1/2)+(1/2)
    src_points *= teeth.shape[1]
    src_points = src_points.astype(np.int32)
    src_points = src_points[teeth_line_start:teeth_line_end]
    src_points = np.expand_dims(src_points, axis=1)
    src_points = np.hstack([src_points, np.zeros_like(src_points)])
    # upper, middle, lower teeth line
    upper_src_points, middle_src_points, lower_src_points = src_points.copy(),src_points.copy(),src_points.copy()
    upper_src_points[:,1] = 0
    middle_src_points[:,1] = int(teeth.shape[0]/2)
    lower_src_points[:,1] = teeth.shape[0]
    src_points = np.vstack([upper_src_points,middle_src_points,lower_src_points])
    # find destination points of transformation
    upper_dst_points = landmarks.get_teeth_line('upper')[teeth_line_start:teeth_line_end].to_XY()
    middle_dst_points = landmarks.get_teeth_line('middle')[teeth_line_start:teeth_line_end].to_XY()
    lower_dst_points = landmarks.get_teeth_line('lower')[teeth_line_start:teeth_line_end].to_XY()
    dst_points = np.vstack([upper_dst_points,middle_dst_points,lower_dst_points])
    # transform teeth img using source points -> destination points mapping
    remaped_teeth = remap_image(teeth, src_points, dst_points) 
    # outside mapping region fill with original image 
    target_image = np.where(remaped_teeth != 0, remaped_teeth, original_img)
    return target_image

def fill_eye_mouth_with_black(img, landmarks):
    poly = [landmarks.get_left_eye_outline().to_XY().astype(int), 
            landmarks.get_right_eye_outline().to_XY().astype(int), 
            landmarks.get_mouth_outline().to_XY().astype(int)]
    img = cv2.fillPoly(img, pts=poly, color=(0, 0, 0))  
    return img

def alter_image(img):
    landmarks = FaceLandmarksDetector().process(img)
    background = add_teeth(img)
    img = fill_eye_mouth_with_black(img, landmarks)
    result_list = []
    frame_rate = 8
    for vectors in tqdm(np.load("motion_vectors.npy")[::int(24/frame_rate)]):
        translated_landmarks = landmarks.translate(vectors)
        remaped_img = remap_image(img, landmarks.get_mesh().to_XY(), translated_landmarks.get_mesh().to_XY())
        remaped_img = fill_eye_mouth_with_black(remaped_img, translated_landmarks)  
        result = np.where(remaped_img != 0, remaped_img, background)
        result_list.append(result)   
    imageio.mimsave('../test/out.gif', result_list, fps=frame_rate)


img = cv2.imread("../test/zeman3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
alter_image(img)