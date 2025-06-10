import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import accuracy_score, classification_report
from evaluate_sigmoid_spec import evaluate_sig

import tensorflow_addons as tfa

X = np.load('data/WhistleDetection/whistle_x_spec_train.npy')  # shape: (samples, freq_bins, time_steps)
y = np.load('data/WhistleDetection/whistle_y_spec_train.npy')

if y.ndim == 2:
    y = y.ravel()


X = X[..., np.newaxis]  # (samples, freq_bins, time_steps, 1)
print(f"Input shape: {X.shape}, Labels: {y.shape}")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2], 1)),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tfa.losses.SigmoidFocalCrossEntropy(gamma=2.0),
              metrics=['accuracy'])
"""
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
"""
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Using GPU?", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    model.fit(X, y,
              batch_size=2048,
              epochs=40,
              callbacks=[callback],)

os.makedirs('model/sigmoid', exist_ok=True)
model.save('model/sigmoid/whistle_model_spec.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model/sigmoid/whistle_model_spec.tflite', 'wb') as f:
    f.write(tflite_model)

evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model_spec.h5")
