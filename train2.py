import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
import glob
from sklearn.metrics import accuracy_score, classification_report


# 加载 .npy 数据
X = np.load('data\\WhistleDetection\\whistle_x_base_train.npy')  # 特征数组
y = np.load('data\\WhistleDetection\\whistle_y_base_train.npy')  # 标签数组

# 如果 y 是二维的，压扁成一维
if y.ndim == 2:
    y = y.ravel()

# 标准化 X
samples, timesteps, features = X.shape
X_reshaped = X.reshape(-1, features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(samples, timesteps, features)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),

    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Using GPU?" , tf.test.is_gpu_available())

#train with GPU
with tf.device('/GPU:0'):
    model.fit(X_scaled, y, batch_size=1024, epochs=20, validation_split=0.2, callbacks=[callback], class_weight={0:1.0, 1:2.0})

model.save('model/whistle_model.h5')



# === 1. 扫描所有模型文件 ===
model_dir = r"D:\Desk\whistle_detection\model\softmax"
model_files = glob.glob(os.path.join(model_dir, "whistle_model*.h5"))
print(model_files)

# === 2. 加载测试数据 ===
X_test = np.load('data\\WhistleDetection\\whistle_x_base_test.npy')
y_test = np.load('data\\WhistleDetection\\whistle_y_base_test.npy')

if y_test.ndim == 2:
    y_test = y_test.ravel()

samples, timesteps, features = X_test.shape
X_test_reshaped = X_test.reshape(-1, features)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(samples, timesteps, features)

# === 3. 遍历并评估每个模型 ===
for model_path in model_files:
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        pred_probs = model.predict(X_test_scaled)
        pred_classes = np.argmax(pred_probs, axis=1)

        acc = accuracy_score(y_test, pred_classes)
        print("file name:", model_path)
        print(f" Accuracy: {acc:.4f}")
        print(classification_report(y_test, pred_classes))

    except Exception as e:
        print(f" Failed to evaluate {model_path}: {e}")
