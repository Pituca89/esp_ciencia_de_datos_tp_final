import cv2
import mediapipe as mp
from datetime import datetime
import argparse
import os
import numpy as np

INPUT_PATH = "data/samples"
# Inicializar el modelo de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del detector de manos
hands = mp_hands.Hands(
    static_image_mode=False,  # Para usar detección en tiempo real
    max_num_hands=2,  # Detectar hasta 2 manos
    min_detection_confidence=0.5,  # Umbral de confianza mínimo para la detección
    min_tracking_confidence=0.5,  # Umbral de confianza mínimo para el seguimiento
)

def execute(**kwargs):
    # Iniciar la cámara
    cap = cv2.VideoCapture(0)
    tag = str(kwargs.get("tag"))
    sample = str(kwargs.get("sample"))
    full_path = INPUT_PATH + "/" +  tag + "/" + sample
    frames = []

    if not os.path.exists(full_path):
        print("Directory not found")
        os.makedirs(full_path, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen de BGR (OpenCV) a RGB (MediaPipe)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen para detectar manos
        results = hands.process(image_rgb)

        # Si se detectan manos, dibujarlas en la imagen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibuja los puntos de referencia y las conexiones de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar la imagen con las manos detectadas
        cv2.imshow("MediaPipe Hands", frame)

        code = cv2.waitKey(1)
        # Capturar imagen al presionar la tecla c
        if code == ord("c"):
            frames.append(np.asarray(frame))
            print("Imagen capturada correctamente")

        # Salir del bucle al presionar la tecla 'q'
        if code == ord("q"):
            break

        # Guardar los frame caputrados con tecla 's'
        if code == ord("s"):
            for idx, iframe in enumerate(frames, 1):
                cv2.imwrite(full_path + "/" + str(idx) + '.jpg', iframe)
            print("Se almacenaron frames en la ruta: ", full_path)

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    execute(**vars(args))