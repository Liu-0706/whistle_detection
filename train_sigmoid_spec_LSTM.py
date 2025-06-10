import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import accuracy_score, classification_report
from evaluate_sigmoid import evaluate_sig

# === Load precomputed spectrograms ===
X = np.load('data/WhistleDetection/whistle_spec_train.npy')
y = np.load('data/WhistleDetection/whistle_spec_train_labels.npy')

if y.ndim == 2:
    y = y.ravel()

# === Add channel dimension ===
X = X[..., np.newaxis]  # (samples, freq, time, 1)
print(f"Loaded input shape: {X.shape}")

# === CNN + LSTM Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2], 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),

    # === Reshape for LSTM ===
    tf.keras.layers.Reshape((-1, 128)),  # (batch, time, features)

    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Using GPU?", tf.config.list_physical_devices('GPU'))

# === Train ===
with tf.device('/GPU:0'):
    model.fit(X, y,
              batch_size=512,
              epochs=50,
              validation_split=0.2,
              callbacks=[callback],
              class_weight={0: 1.0, 1: 2.0})

# === Save model and convert to TFLite ===
os.makedirs('model/sigmoid', exist_ok=True)
model.save('model/sigmoid/whistle_model_lstm.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model/sigmoid/whistle_model_lstm.tflite', 'wb') as f:
    f.write(tflite_model)

# === Evaluate ===
evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model_lstm.h5")
