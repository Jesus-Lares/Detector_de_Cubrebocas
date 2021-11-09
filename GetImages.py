import cv2
import mediapipe as mp
import os

direction = "E:/7_semestre/vison_artificial/Detector_de_Cubrebocas/Images"
name="Without_mask"
folder = f"{direction}/{name}"

if not os.path.exists(folder):
    os.makedirs(folder)

mp_face_detection = mp.solutions.face_detection
cap=cv2.VideoCapture(0)
count=0

with mp_face_detection.FaceDetection(min_detection_confidence=.5) as face_detection:
    while True:
        ret,frame = cap.read()
        if ret == False: break

        height,width,_ = frame.shape
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections is not None:
            for detection in results.detections:
                xmin = int(detection.location_data.relative_bounding_box.xmin*width)
                ymin = int(detection.location_data.relative_bounding_box.ymin*height)
                w = int(detection.location_data.relative_bounding_box.width*width)
                h = int(detection.location_data.relative_bounding_box.height*height)
                if xmin <0 and ymin<0:
                    continue
                face_image = frame[ymin:ymin+h,xmin:xmin+w]
                face_rec = cv2.resize(face_image,(150,200),interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(f"{folder}/face_{str(count)}.jpg",face_rec)
                count+=1                
              
        cv2.imshow("Entrenamiento",frame)
        
        t=cv2.waitKey(1)
        if t==27 or count >=300:
            break
            
cap.release()
cv2.destroyAllWindows() 