import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np

# Definir rutas
INPUT_PATH = "data/samples"
OUTPUT_PATH = "data/model"

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def extract_keypoints(image_path):
    """Extrae los keypoints de una imagen usando MediaPipe Hands."""
    frame = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            return keypoints / np.linalg.norm(keypoints)

    # Si no se detecta una mano, se llena con ceros
    return np.zeros(21 * 3)

def process_samples():
    """Recorre todas las carpetas de samples y almacena los keypoints en un archivo HDF5."""
    tags = os.listdir(INPUT_PATH)  # Obtener todas las etiquetas (nombres de las carpetas)

    for tag in tags:
        tag_path = os.path.join(INPUT_PATH, tag)
        if not os.path.isdir(tag_path):
            continue  # Ignorar si no es una carpeta

        data = []  # Lista para almacenar las secuencias de keypoints
        sample_folders = sorted(os.listdir(tag_path))  # Ordena para mantener secuencia temporal

        for sample_id, sample_folder in enumerate(sample_folders, start=1):
            sample_path = os.path.join(tag_path, sample_folder)
            if not os.path.isdir(sample_path):
                continue  # Ignorar archivos sueltos

            frame_files = sorted(os.listdir(sample_path))  # Ordena imágenes en la secuencia temporal
            keypoint_seq = [extract_keypoints(os.path.join(sample_path, frame)) for frame in frame_files]

            # Agregar la secuencia de keypoints con su etiqueta y sample ID
            for frame_idx, keypoints in enumerate(keypoint_seq, start=1):
                data.append([tag, sample_id, frame_idx, keypoints])

        # Convertir los datos en DataFrame
        df = pd.DataFrame(data, columns=["tag", "sample", "frame", "keypoints"])

        # Crear carpeta de salida si no existe
        os.makedirs(f"{OUTPUT_PATH}/hdf", exist_ok=True)

        # Guardar en HDF5
        output_file = f"{OUTPUT_PATH}/hdf/{tag}.h5"
        df.to_hdf(output_file, key="dataframe", index=False, mode="w")

        print(f"Guardado: {output_file} ({len(df)} muestras)")

if __name__ == '__main__':
    process_samples()
