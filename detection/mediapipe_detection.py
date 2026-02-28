import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

class DetectorManosMediaPipe:
    def __init__(self, model_path, num_hands = 2, running_mode = 'IMAGE'):
        mode = getattr(mp.tasks.vision.RunningMode, running_mode)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=num_hands, running_mode=mode)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detectarManos(self, imagenCV2):
        imagenRGB = cv2.cvtColor(imagenCV2, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagenRGB)
        detection_result = self.detector.detect(imagen)
        return detection_result

    def obtenerBbox(self, imagenCV2, detection_result):
        if not detection_result.hand_landmarks:
            return None

        alto, ancho, _ = imagenCV2.shape
        xs = []
        ys = []

        for mano in detection_result.hand_landmarks:
            for land in mano:  # MediaPipe entrega coordenadas normalizadas
                xs.append(land.x * ancho)
                ys.append(land.y * alto)

        xMin, xMax = min(xs), max(xs)
        yMin, yMax = min(ys), max(ys)
        w = xMax - xMin
        h = yMax - yMin
        return xMin, yMin, w, h  # x, y, w, h


    def recortarImagen(self, imagen, bbox, tamImagen=(256, 256)):
        x, y, w, h = bbox

        centroX = x + (w / 2)
        centroY = y + (h / 2)

        # Hacer el cuadrado un poco más grande
        lado = max(w, h) * 1.2  # Margen del 20%
        mitad = lado / 2

        src_pts = np.array([
            [centroX - mitad, centroY - mitad],  # Arriba izq
            [centroX - mitad, centroY + mitad],  # Abajo izq
            [centroX + mitad, centroY - mitad],  # Arriba der
        ], dtype=np.float32)

        dst_pts = np.array([
            [0, 0],  # Arriba izq
            [0, tamImagen[1]],  # Abajo izq
            [tamImagen[0], 0],  # Arriba der
        ], dtype=np.float32)

        matriz = cv2.getAffineTransform(src_pts, dst_pts)

        imagenRecortada = cv2.warpAffine(imagen, matriz, tamImagen, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return imagenRecortada, (int(centroX - mitad), int(centroY - mitad), int(lado), int(lado)) # imagen, bbox ampliada

    def mostrarResultados(self, imagenCV2, bboxAmpliada, imagenRecortada):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imagenCV2, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        img_bbox = imagenCV2.copy()
        x, y, w, h = bboxAmpliada
        cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 5)
        plt.imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(imagenRecortada, cv2.COLOR_BGR2RGB))

        plt.show()



modelPath = r"C:\Users\jalex\Downloads\hand_landmarker.task"

webcam = True
if not webcam:
    rutaImagen = r"D:\download\m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands\Mugsy_cameras\envmap_per_frame\images\400266\161415.png"

    imagenCV2 = cv2.imread(rutaImagen)

    detector = DetectorManosMediaPipe(model_path = modelPath, num_hands=2, running_mode='IMAGE')

    deteccion = detector.detectarManos(imagenCV2)
    bbox = detector.obtenerBbox(imagenCV2, deteccion)

    if bbox is not None:
        imagenRecortada, bboxAmpliada = detector.recortarImagen(imagenCV2, bbox, tamImagen=(256, 256))
        detector.mostrarResultados(imagenCV2, bboxAmpliada, imagenRecortada)
    else:
        print("No se detectaron manos")

else:
    detector = DetectorManosMediaPipe(modelPath, 2, 'IMAGE')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        exito, frame = cap.read() #frame esta en BGR

        frame = cv2.flip(frame, 1) # Para que se vea modo espejo

        deteccion = detector.detectarManos(frame)
        bbox = detector.obtenerBbox(frame, deteccion)

        if bbox is not None:
            imagenRecortada, bboxAmpliada = detector.recortarImagen(frame, bbox, tamImagen=(256, 256))
            x, y, w, h = bboxAmpliada
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Mano detectada', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Imagen recortada 256x256',imagenRecortada)

        cv2.imshow('Webcam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()