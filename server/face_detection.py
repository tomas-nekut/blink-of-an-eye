
import face_recognition
import cv2

image = face_recognition.load_image_file("../test/zeman2.jpeg")


face_locations = face_recognition.face_locations(image)[0]


#face = image[face_locations[0]:face_locations[0]+face_locations[2], face_locations[1]:face_locations[1]+face_locations[3]]
#cv2.imwrite("id_test.jpg", face)


known_image = face_recognition.load_image_file("../test/zeman2.jpeg")
unknown_image = face_recognition.load_image_file("../test/tim.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

print(biden_encoding)
print(unknown_encoding)
print(results)