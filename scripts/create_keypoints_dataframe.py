import cv2
import mediapipe as mp
from datetime import datetime
import argparse
import os
import pandas as pd
import numpy as np

INPUT_PATH = "data/samples"
OUTPUT_PATH = "data/model"
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

def insert_keypoints_sequence(df, n_sample:int, kp_seq):
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints], ignore_index=True)
    return df

def get_keypoint_seq(path_image):
    kp_seq = np.array([])
    for image_name in os.listdir(path_image):
        image_path = os.path.join(path_image, image_name)
        frame = cv2.imread(image_path)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                kp_frame = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        else:
            kp_frame = np.zeros(21*3)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    print("Path: " + path_image + " Size: " + str(kp_seq.size))
    return kp_seq

def execute(**kwargs):
    tags = kwargs.get("tags")

    for tag in tags:
        data = pd.DataFrame([])
        path_frames = INPUT_PATH + "/" +  tag
        for idx, image in enumerate(os.listdir(path_frames), 1):
            full_path_frames = path_frames + "/" + image
            keypoint_sequence = get_keypoint_seq(full_path_frames)
            data_temp = insert_keypoints_sequence(data, idx, keypoint_sequence)
            data = pd.concat([data, data_temp], ignore_index=True)

        if not os.path.exists(f"{OUTPUT_PATH}/hdf"):
            os.makedirs(f"{OUTPUT_PATH}/hdf", exist_ok=True)

        data.to_hdf(f"{OUTPUT_PATH}/hdf/{tag}.h5", key='dataframe', index=False, mode="w")
        #data.to_csv(f"{OUTPUT_PATH}/hdf/{tag}.csv", index=False, mode="w")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    args = parser.parse_args()

    execute(**vars(args))