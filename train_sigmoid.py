import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
from sklearn.metrics import accuracy_score, classification_report
from evaluate_sigmoid import evaluate_sig

X = np.load('data\\WhistleDetection\\whistle_x_base_train.npy')
y = np.load('data\\WhistleDetection\\whistle_y_base_train.npy')

if y.ndim == 2:
    y = y.ravel()

samples, timesteps, features = X.shape
X_reshaped = X.reshape(-1, features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(samples, timesteps, features)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),

    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                         patience=10, 
                                         restore_best_weights=True)

print("Using GPU?", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    model.fit(X_scaled, y,
              batch_size=1024,
              epochs=50,
              callbacks=[callback],)

os.makedirs('model/sigmoid', exist_ok=True)
model.save('model/sigmoid/whistle_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model/sigmoid/whistle_model.tflite', 'wb') as f:
    f.write(tflite_model)


evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model.h5")


"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
from sklearn.metrics import accuracy_score, classification_report
from evaluate_sigmoid import evaluate_sig

X = np.load('data\\WhistleDetection\\whistle_x_base_train.npy')
y = np.load('data\\WhistleDetection\\whistle_y_base_train.npy')

if y.ndim == 2:
    y = y.ravel()

samples, timesteps, features = X.shape
X_reshaped = X.reshape(-1, features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(samples, timesteps, features)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),

    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Using GPU?", tf.config.list_physical_devices('GPU'))

#train
with tf.device('/GPU:0'):
    model.fit(X_scaled, y,
              batch_size=512,
              epochs=50,
              validation_split=0.2,
              callbacks=[callback],
              class_weight={0: 1.0, 1: 2.0})

# === Save model (.h5 and .tflite) ===
os.makedirs('model/sigmoid', exist_ok=True)
model.save('model/sigmoid/whistle_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model/sigmoid/whistle_model.tflite', 'wb') as f:
    f.write(tflite_model)

evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model.h5")
"""