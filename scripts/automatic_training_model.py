import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Paths y constantes
KEY_POINTS_PATH = "data/model/hdf"
MAX_MODEL_FRAMES = 30
LENGTH_KEYPOINTS = 63
OUTPUT_MODEL = 'models/hands_detection.keras'

def pad_or_truncate_sequence(seq, max_length):
    """Asegura que todas las secuencias tengan la misma longitud."""
    if len(seq) < max_length:
        padding = [np.zeros_like(seq[0])] * (max_length - len(seq))
        seq = seq + padding
    else:
        seq = seq[:max_length]
    return np.array(seq)

def get_tags():
    """Obtiene automáticamente las etiquetas desde los archivos .h5 disponibles."""
    tags = [f.replace(".h5", "") for f in os.listdir(KEY_POINTS_PATH) if f.endswith(".h5")]
    return sorted(tags)  # Ordena para mantener consistencia

def get_sequences_and_labels(tags):
    sequences, labels = [], []
    for idx_tag, tag in enumerate(tags):
        hdf_path = os.path.join(KEY_POINTS_PATH, f"{tag}.h5")
        data = pd.read_hdf(hdf_path, key="dataframe")
        
        for _, df_sample in data.groupby('sample'):
            seq_keypoints = [fila['keypoints'] for _, fila in df_sample.iterrows()]
            if len(seq_keypoints) >= MAX_MODEL_FRAMES // 2:
                seq_keypoints = pad_or_truncate_sequence(seq_keypoints, MAX_MODEL_FRAMES)
                sequences.append(seq_keypoints)
                labels.append(idx_tag)
    return sequences, labels

def execute():
    # Obtener etiquetas automáticamente
    tags = get_tags()
    print(f"Detectadas etiquetas: {tags}")

    sequences, labels = get_sequences_and_labels(tags)
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42, shuffle=True)
    early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(MAX_MODEL_FRAMES, LENGTH_KEYPOINTS)),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),  # Última capa sin return_sequences
        Dropout(0.3),
        Dense(len(tags), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=8, callbacks=[early_stopping])

    model.summary()
    model.save(OUTPUT_MODEL)

if __name__ == '__main__':
    execute()
