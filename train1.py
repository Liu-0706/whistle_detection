import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# 加载 .npy 数据
X = np.load('data\WhistleDetection\whistle_x_base_train.npy')  # 例如 shape: (1000, 99, 10)
y = np.load('data\WhistleDetection\whistle_y_base_train.npy')  # 例如 shape: (1000,) 或 (1000, 1)

# 如果 y 是二维的，压扁成一维
if y.ndim == 2:
    y = y.ravel()

# 标准化 X（按特征维度）
samples, timesteps, features = X.shape
X_reshaped = X.reshape(-1, features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(samples, timesteps, features)

# 构建模型（可用你之前优化过的版本）
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')  # 如果是二分类，也可用 sigmoid+1输出

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping 回调
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Using GPU?" , tf.test.is_gpu_available())

# 训练
with tf.device('/GPU:0'):
    model.fit(X_scaled, y, batch_size=32, epochs=20, validation_split=0.2, callbacks=[callback])

# 保存模型
model.save('model/whistle_model.h5')