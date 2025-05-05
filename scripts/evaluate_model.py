import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

MAX_MODEL_FRAMES = 30
LENGTH_KEYPOINTS = 63
OUTPUT_MODEL = 'models/hands_detection.keras'
TAG_NAMES = [
    "moving_close_left_hand",
    "moving_close_rigth_hand",
    "moving_open_left_hand",
    "moving_open_rigth_hand"
]

# Cargar el modelo entrenado
model = load_model(OUTPUT_MODEL)

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Para dibujar landmarks y conexiones

# Almacenar frames de la secuencia actual
current_sequence = []
last_detected_label = "No reconocida"

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    predictions = None  # Inicializar la variable
    predicted_class = None
    # Convertir la imagen de BGR a RGB y procesarla con MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dibujar los landmarks y el recuadro si se detecta una mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calcular las coordenadas del bounding box (recuadro)
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Dibujar el recuadro
            cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            # Extraer puntos clave si se detecta una mano
            keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()

            # Asegurar que los keypoints tengan el tamaño correcto
            if len(keypoints) != LENGTH_KEYPOINTS:
                keypoints = np.pad(keypoints, (0, LENGTH_KEYPOINTS - len(keypoints)), mode='constant')

            # Añadir puntos clave a la secuencia actual
            current_sequence.append(keypoints)

            # Mantener el tamaño de la secuencia (MAX_MODEL_FRAMES)
            if len(current_sequence) > MAX_MODEL_FRAMES:
                current_sequence.pop(0)

            # Si la secuencia alcanza el tamaño máximo, hacer la predicción
            if len(current_sequence) == MAX_MODEL_FRAMES:
                # Rellenar con ceros si hay menos de MAX_MODEL_FRAMES
                while len(current_sequence) < MAX_MODEL_FRAMES:
                    current_sequence.append(np.zeros(LENGTH_KEYPOINTS))

                # Convertir la secuencia en formato numpy array
                sequence_array = np.array([current_sequence])

                # Obtener predicción del modelo
                predictions = model.predict(sequence_array)
                predicted_class = np.argmax(predictions)

                # # Aplicar umbral de confianza del 80% y evitar fluctuaciones
                if predictions[0][predicted_class] > 0.8:
                    new_label = TAG_NAMES[predicted_class]
                else:
                    new_label = last_detected_label  # Mantener la etiqueta anterior si la confianza es baja

                # Solo actualizar el texto en pantalla si ha cambiado la detección
                last_detected_label = new_label
    else:
        # Si no se detecta ninguna mano, vaciar la secuencia
        current_sequence = []
        last_detected_label = "No reconocida"

    # Mostrar el último gesto detectado en pantalla con confianza
    confidence = predictions[0][predicted_class] * 100 if predictions is not None else 0
    cv2.putText(image, f"Label: {last_detected_label} ({confidence:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame procesado
    cv2.imshow('Detección de señas', image)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()