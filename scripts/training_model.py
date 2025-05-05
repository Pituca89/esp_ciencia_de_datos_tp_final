import argparse
import os
import pandas as pd
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D #AveragePooling2D
from tensorflow.keras import backend as K
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from collections import Counter

KEY_POINTS_PATH = "data/model/hdf"
MAX_MODEL_FRAMES = 16
LENGTH_KEYPOINTS = 63
OUTPUT_LENGTH = 3
OUTPUT_MODEL = 'models/hands_detection.keras'

def pad_or_truncate_sequence(seq, max_length):
    """Asegura que todas las secuencias tengan la misma longitud."""
    if len(seq) < max_length:
        # Si la secuencia es más corta, rellena con ceros
        padding = [np.zeros_like(seq[0])] * (max_length - len(seq))
        seq = seq + padding
    else:
        # Si es más larga, trunca
        seq = seq[:max_length]
    return np.array(seq)

def get_sequences_and_labels(tags):
    sequences, labels = [], []
    for idx_tag, tag in enumerate(tags):
        hdf_path = os.path.join(KEY_POINTS_PATH, f"{tag}.h5")
        data = pd.read_hdf(hdf_path, key="dataframe")
        
        for _, df_sample in data.groupby('sample'):
            seq_keypoints = [fila['keypoints'] for _, fila in df_sample.iterrows()]
            seq_keypoints = pad_or_truncate_sequence(seq_keypoints, MAX_MODEL_FRAMES)
            sequences.append(seq_keypoints)
            labels.append(idx_tag)
    print("FIN: ", len(seq_keypoints))
    return sequences, labels

def execute(**kwargs):
    tags = kwargs.get("tags")
    sequences, labels = get_sequences_and_labels(tags)

    X = np.array(sequences)
    y = to_categorical(labels).astype(int) 
    early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True)

    model = Sequential()
    
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(MAX_MODEL_FRAMES, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(OUTPUT_LENGTH, activation='softmax'))

    # optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, callbacks=[early_stopping])
    
    model.summary()
    model.save(OUTPUT_MODEL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    args = parser.parse_args()

    execute(**vars(args))