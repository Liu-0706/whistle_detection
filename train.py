import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 加载数据
data = np.load('dataset.npz', allow_pickle=True)
X = np.array(data['X'].tolist(), dtype=np.float32)
y = np.array(data['y'], dtype=np.int32)

# 标准化输入
samples, timesteps, features = X.shape
X_reshaped = X.reshape(-1, features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped).reshape(samples, timesteps, features)

# 构建改进模型
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
    tf.keras.layers.Dense(2, activation='softmax')  # 若为二分类且标签为0/1，也可改为 Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 若输出为 sigmoid，则改为 binary_crossentropy
              metrics=['accuracy'])

# 训练模型，加入 EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.summary()
model.fit(X_scaled, y, batch_size=16, epochs=200, validation_split=0.2, callbacks=[callback])

# 保存模型
model.save('model/whistle_model.h5')
