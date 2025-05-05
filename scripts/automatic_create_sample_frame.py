import cv2
import mediapipe as mp
import os
import time
import argparse

# Ruta donde se guardarán los samples
INPUT_PATH = "data/samples"
COUNT_FRAMES = 30

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del detector de manos
hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=2,  
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5,  
)

def create_sample_folder(tag):
    """ Crea una nueva carpeta para un sample cuando detecta una nueva mano """
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Nombre con fecha y hora
    sample_path = os.path.join(INPUT_PATH, tag, f"sample_{timestamp}")
    os.makedirs(sample_path, exist_ok=True)
    return sample_path

def execute(tag):
    cap = cv2.VideoCapture(0)
    
    capturing = False  # Estado de captura
    sample_path = None  # Ruta actual del sample
    frame_count = 0  # Contador de imágenes
    last_capture_time = 0  # Última vez que se capturó un frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir a RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Detecta si hay manos
        hand_detected = results.multi_hand_landmarks is not None

        if hand_detected:
            # Dibujar los landmarks de la mano en la imagen
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Si antes no estaba capturando, crear una nueva carpeta
            if not capturing:
                sample_path = create_sample_folder(tag)
                print(f"Iniciando nuevo sample en {sample_path}")
                frame_count = 0
                capturing = True

            # Captura cada 1 segundo
            current_time = time.time()
            if current_time - last_capture_time >= 0.1 and frame_count < COUNT_FRAMES:
                frame_count += 1
                filename = os.path.join(sample_path, f"{frame_count}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Imagen guardada: {filename}")
                last_capture_time = current_time

        else:
            # Si la mano desaparece, detiene la captura
            if capturing:
                print("Mano fuera de pantalla. Finalizando sample.")
                capturing = False

        # Mostrar la imagen con los landmarks
        cv2.imshow("MediaPipe Hands", frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()
    execute(args.tag)
