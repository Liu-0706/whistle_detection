import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

def evaluate_sig(model_path):
    # === Load trained model ===
    model = tf.keras.models.load_model(model_path, compile=False)
    X_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_x_spec_test.npy')
    y_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_y_spec_test.npy')

    if y_test.ndim == 2:
        y_test = y_test.ravel()

    X_test = X_test[..., np.newaxis]  # shape: (samples, freq, time, 1)

    # === Predict ===
    pred_probs = model.predict(X_test)

    if pred_probs.shape[1] == 1:
        prob_whistle = pred_probs[:, 0]
        prob_not_whistle = 1 - prob_whistle
    else:
        prob_not_whistle = pred_probs[:, 0]
        prob_whistle = pred_probs[:, 1]

    # You can adjust threshold (0.95 is stricter than 0.5)
    pred_labels = (prob_whistle >= 0.95).astype(int)

    # === Save detailed results ===
    df = pd.DataFrame({
        'sample_index': np.arange(len(y_test)),
        'prob_not_whistle': prob_not_whistle,
        'prob_whistle': prob_whistle,
        'predicted_label': pred_labels,
        'true_label': y_test
    })

    df.to_csv('whistle_probabilities_with_labels.csv', index=False)
    print("Saved as 'whistle_probabilities_with_labels.csv'")

    # === Print metrics ===
    accuracy = accuracy_score(y_test, pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, pred_labels))


#evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model_spec.h5")