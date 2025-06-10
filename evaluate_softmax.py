import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# === 1. 扫描所有模型文件 ===
model_dir = r"D:\Desk\whistle_detection\model\softmax"
model_files = glob.glob(os.path.join(model_dir, "whistle_model*"))
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
        print("\n=== Evaluating:", model_path)

        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path, compile=False)
            pred_probs = model.predict(X_test_scaled)
            pred_classes = np.argmax(pred_probs, axis=1)

        elif model_path.endswith('.tflite'):
            # 加载 TensorFlow Lite 模型
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
