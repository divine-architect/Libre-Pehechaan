from imageai.Detection import ObjectDetection
import os
import easyocr
import pyautogui
import cv2
import face_recognition
from blink import main

execution_path = os.getcwd()

def person_detect_in_id():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , r"yolov3.pt"))
    detector.loadModel()

    detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , r"Untitled.png"), output_image_path=os.path.join(execution_path , "image4new.jpg"), minimum_percentage_probability=30,  extract_detected_objects=True)

    for eachObject, eachObjectPath in zip(detections, objects_path):
       # print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )

        if eachObject['percentage_probability'] > 90:
            return "Picture found"
        else:
            return "Picutre not found. Invalid ID"

def ocrfunc():
    img = cv2.imread('Untitled.png')
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img, detail = 0)

    imp_info = []
    for i in result:
        if i.startswith('Name:'):
            name = i[5:]
            imp_info.append(name.strip())
        if i.startswith('Age:'):
            age = i[4:]
            imp_info.append(age.strip())
        if i.startswith('DOB:'):
            dob = i[4:]
            imp_info.append(dob.strip())
        
    return imp_info

def take_photo():
    
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
             
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def comparefaces():
    known_image = face_recognition.load_image_file(os.path.join(execution_path , "image4new.jpg"))
    unknown_image = face_recognition.load_image_file("bane.png")

    biden_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    return results

print("Detecting person")
print(person_detect_in_id())
print("Fetching info from ID")
#print(ocrfunc())
print("Press space bar to take a photo, press esc to close")
take_photo()
print("Detecting if person is alive")
main()
print("Comparing faces")
print(comparefaces())
