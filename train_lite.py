import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
import glob
from sklearn.metrics import accuracy_score, classification_report

X = np.load('data\\WhistleDetection\\whistle_x_base_train.npy')
y = np.load('data\\WhistleDetection\\whistle_y_base_train.npy')

# if y is 2D
if y.ndim == 2:
    y = y.ravel()

# Normalize x
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
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Using GPU?", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    model.fit(X_scaled, y, batch_size=1024, epochs=40, callbacks=[callback], class_weight={0:1.0, 1:2.0})
    #model.fit(X_scaled, y, batch_size=2048, epochs=100, validation_split=0.2, class_weight={0:1.0, 1:2.0})
    """
    model.fit(X_scaled, y, batch_size=2048, epochs=50, validation_split=0.2, 
              callbacks=[callback], class_weight={0:1.0, 1:2.0})
    """
os.makedirs('model', exist_ok=True)
model.save('model/sigmoid/whistle_model.h5')



converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()


tflite_model_path = 'model/whistle_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)



# ====================================================
model_dir = r"D:\Desk\whistle_detection\model\softmax"
model_files = glob.glob(os.path.join(model_dir, "whistle_model*"))
print(model_files)

X_test = np.load('data\\WhistleDetection\\whistle_x_base_test.npy')
y_test = np.load('data\\WhistleDetection\\whistle_y_base_test.npy')

if y_test.ndim == 2:
    y_test = y_test.ravel()

samples, timesteps, features = X_test.shape
X_test_reshaped = X_test.reshape(-1, features)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(samples, timesteps, features)


for model_path in model_files:
    try:
        print("\n=== Evaluating:", model_path)

        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path, compile=False)
            pred_probs = model.predict(X_test_scaled)
            pred_classes = np.argmax(pred_probs, axis=1)

        elif model_path.endswith('.tflite'):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            pred_classes = []
            for i in range(X_test_scaled.shape[0]):
                sample = X_test_scaled[i:i+1].astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], sample)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                pred_classes.append(np.argmax(output_data))

        else:
            print(f"Unsupported file format: {model_path}")
            continue

        acc = accuracy_score(y_test, pred_classes)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, pred_classes))

    except Exception as e:
        print(f"Failed to evaluate {model_path}: {e}")

