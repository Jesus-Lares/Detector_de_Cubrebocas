import cv2
import os
import numpy as np

direction = "E:/7_semestre/vison_artificial/Detector_de_Cubrebocas/Images/Train"
listImages = os.listdir(direction)

labels = []
faces = []
count = 0

for nameDir in listImages:
    name = direction + "/" + nameDir
    
    for fileName in os.listdir(name):
        image_path= name + "/" + fileName
        print(image_path)
        image=cv2.imread(image_path,0)
        labels.append(count)
        faces.append(image)
        
        #cv2.imshow("image",image)
        #cv2.waitKey(10)
    count += 1
    
print("Etiqueta 0: ",np.count_nonzero(np.array(labels)==0)," con cubrebocas")
print("Etiqueta 1: ",np.count_nonzero(np.array(labels)==1)," sin cubrebocas")

recognition = cv2.face.LBPHFaceRecognizer_create()
recognition.train(faces,np.array(labels))

recognition.write("./Models/ModelTrain.xml")
print("Modelo Creado")