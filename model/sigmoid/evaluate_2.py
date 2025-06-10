import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def evaluate_sigmoid(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)

    X_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_x_base_test.npy')
    y_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_y_base_test.npy')

    if y_test.ndim == 2:
        y_test = y_test.ravel()

    samples, timesteps, features = X_test.shape

    X_test_reshaped = X_test.reshape(-1, features)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(samples, timesteps, features)

    pred_probs = model.predict(X_test_scaled)

    if pred_probs.shape[1] == 1:
        # sigmoid
        prob_whistle = pred_probs[:, 0]
        prob_not_whistle = 1 - prob_whistle
    else:   
        # softmax
        prob_not_whistle = pred_probs[:, 0]
        prob_whistle = pred_probs[:, 1]

    pred_labels = (prob_whistle >= 0.95).astype(int)

    df = pd.DataFrame({
        'sample_index': np.arange(len(y_test)),
        'prob_not_whistle': prob_not_whistle,
        'prob_whistle': prob_whistle,
        'predicted_label': pred_labels,
        'true_label': y_test
    })

    df.to_csv('whistle_probabilities_with_labels.csv', index=False)
    print("save as 'whistle_probabilities_with_labels.csv'")

    accuracy = accuracy_score(y_test, pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, pred_labels))


evaluate_sigmoid('whistle_model_0.88.h5')