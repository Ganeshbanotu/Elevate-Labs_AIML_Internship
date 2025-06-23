import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
def load_data(data_dir, img_size=100):
    X = []
    y = []
    labels = {"with_mask": 0, "without_mask": 1}

    for label, idx in labels.items():
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(idx)

    X = np.array(X) / 255.0  # Normalize
    y = to_categorical(y)    # One-hot encode labels
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main
if __name__ == "__main__":
    data_path = "dataset"
    X_train, X_val, y_train, y_val = load_data(data_path)

    model = build_model(X_train.shape[1:])
    model.summary()

    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

    model.save("model/mask_detector_model.h5")
    print("Model saved to model/mask_detector_model.h5")
