import uuid
from face_animation import FaceAnimator

face_animator = FaceAnimator()
src_path = "../test/zeman3.jpg" 
dst_path = str(uuid.uuid4()) + ".png"
animation_successful = face_animator.process(src_path, dst_path)