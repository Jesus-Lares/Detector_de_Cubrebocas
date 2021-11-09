# Detector_de_Cubrebocas

Detección de cubrebocas usando openCV y mediaPipe

El proyecto se divide en 3 partes:

- Obtención de imagenes.
- Creación del modelo.
- Detección de cubrebocas.

## Obtención de imagenes

Para obtener las imagenes tenemos dos metodos: Buscar un dataset o crearlo. Se decidio por buscar el dataset y en la busqueda se encontro un [dataset de kaggle](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset?select=Face+Mask+Dataset) usando la carpeta Test ya que contaba con las suficientes imagenes para tener un buen modelo y no tenia tantas para que fuera muy pesado. Por problematicas futuras, se creo el archivo de [GetImages.py](https://github.com/Jesus-Lares/Detector_de_Cubrebocas/blob/master/GetImages.py) para obtener 300 fotos dependiendo de lo que pidieras para tener un mejor modelo.
Estas imágenes fueron obtenidas luego de aplicar la detección de rostros con mediapipe, para ser redimensionadas con openCV a cierto alto y ancho.

## Creación del modelo

Ya que se tiene el dataset, se entrena el modelo con la funcion cv2.face.LBPHFaceRecognizer_create() que es uno de los métodos para el reconocimiento facial, para esto se creo el archivo de [CreateModel.py](https://github.com/Jesus-Lares/Detector_de_Cubrebocas/blob/master/CreateModel.py).

## Detección de cubrebocas

Una vez llegado a este punto se procede a realizar las pruebas en un video de tiempo real usando nuevamente la libreria mediaPipe y la funcion cv2.face.LBPHFaceRecognizer_create(). Se creo el archivo de [DetectMask.py](https://github.com/Jesus-Lares/Detector_de_Cubrebocas/blob/master/DetectMask.py), en este punto, se tuvo la problematica puesto que se tenia una camara de baja calidad y poca luminosidad, por lo que, se opto por modificar el dataset agregando imagenes de nosotros con el cubrebocas y sin el. Haciendo esto el modelo funciono a la perfección.
