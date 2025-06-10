import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# === 加载模型 ===
model_path = 'whistle_model_0.84.h5'
model = tf.keras.models.load_model(model_path, compile=False)

X_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_x_base_test.npy')
y_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_y_base_test.npy')

if y_test.ndim == 2:
    y_test = y_test.ravel()

samples, timesteps, features = X_test.shape

# === 标准化 ===
X_test_reshaped = X_test.reshape(-1, features)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(samples, timesteps, features)

# === 模型预测 ===
pred_probs = model.predict(X_test_scaled)

# === 判断输出类型 ===
if pred_probs.shape[1] == 1:
    # sigmoid输出（概率是哨声）
    prob_whistle = pred_probs[:, 0]
    prob_not_whistle = 1 - prob_whistle
else:
    # softmax输出（分别给出每类概率）
    prob_not_whistle = pred_probs[:, 0]
    prob_whistle = pred_probs[:, 1]

# === 判断模型预测的标签（哪个概率大就判为哪类）===
pred_labels = (prob_whistle >= prob_not_whistle).astype(int)

# === 构建 DataFrame 并保存 ===
df = pd.DataFrame({
    'sample_index': np.arange(len(y_test)),
    'prob_not_whistle': prob_not_whistle,
    'prob_whistle': prob_whistle,
    'predicted_label': pred_labels,
    'true_label': y_test
})

df.to_csv('whistle_probabilities_with_labels.csv', index=False)
print("save as 'whistle_probabilities_with_labels.csv'")