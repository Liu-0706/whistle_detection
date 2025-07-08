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

#if y.ndim == 2:
    #y = y.ravel()

samples, timesteps, features = X.shape
X_reshaped = X.reshape(-1, features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(samples, timesteps, features)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
    #tf.keras.layers.Input(shape=X.shape),
    #tf.keras.layers.Input(shape=(513, 1)),

    #1
    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.3),

    #2
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.3),

    #3
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.3),

    #tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dropout(0.3),

    #tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='loss', 
                                         patience=10, 
                                         restore_best_weights=True)

print("Using GPU?", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    model.fit(X_scaled, y,
              batch_size=512,
              epochs=2,
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
#from evaluate_sigmoid import evaluate_sig

X = np.load('data\\WhistleDetection\\whistle_x_base_train.npy')
y = np.load('data\\WhistleDetection\\whistle_y_base_train.npy')

#(544503, 1) -> (544503,)
if y.ndim == 2:
    y = y.ravel()

X_small = X[:400000]
print("X_small.shape",X_small.shape)
y_small = y[:400000]
"""
# === Normalize X ===
samples, timesteps, features = X.shape
X_reshaped = X.reshape(-1, features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(samples, timesteps, features)
print("X_scaled.shape",X_scaled.shape)
"""

"""
# === Define sigmoid model for binary classification ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),

    tf.keras.layers.Conv1D(32, 5, strides=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(2, strides=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(64, 5, strides=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(2, strides=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(128, 5, strides=2, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""

model = tf.keras.Sequential([
    #tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
    tf.keras.layers.Input(shape=(513, 1)),
    
    tf.keras.layers.Conv1D(32, 5, strides=2, activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPooling1D(2, strides=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(64, 5, strides=2, activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPooling1D(2, strides=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(128, 5, strides=2, activation=tf.nn.leaky_relu),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation=tf.keras.activations.elu),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',  
              metrics=['accuracy'])


callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

print("Using GPU?", tf.config.list_physical_devices('GPU'))
""" 
with tf.device('/GPU:0'):
    model.fit(X_scaled, y,
              batch_size=512,
              epochs=40,
              callbacks=[callback],
              class_weight={0: 1.0, 1: 2.0})
"""

with tf.device('/GPU:0'):
    model.fit(X_small, y_small,
              batch_size=256,
              epochs=50,
              callbacks=[callback],
              class_weight={0: 1.0, 1: 2.0})

os.makedirs('model/sigmoid', exist_ok=True)
model.save('model/sigmoid/whistle_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model/sigmoid/whistle_model.tflite', 'wb') as f:
    f.write(tflite_model)

###
#evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model.h5")

