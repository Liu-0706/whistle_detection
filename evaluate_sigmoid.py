import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def evaluate_sig(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
<<<<<<< HEAD

    x_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_x_base_test.npy')
    y_test = np.load('D:\\Desk\\whistle_detection\\data\\WhistleDetection\\whistle_y_base_test.npy')

    #if y_test.ndim == 2:
        #y_test = y_test.ravel()
        #y_test = y_test[:,0]

    #x_test = x_test[:,0]

    #samples, timesteps, features = X_test.shape

    #X_test_reshaped = X_test.reshape(-1, features)
    #scaler = StandardScaler()
    #X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(samples, timesteps, features)



    pred_probs = model.predict(x_test)
    #print(pred_probs)

    '''
    if pred_probs.shape[1] == 1:
        # sigmoid
        prob_whistle = pred_probs[:, 0]
        prob_not_whistle = 1 - prob_whistle
    else:   
        # softmax
        prob_not_whistle = pred_probs[:, 0]
        prob_whistle = pred_probs[:, 1]
    '''

    pred_labels = (pred_probs >= 0.95).astype(int)
    err = pred_labels - y_test
    print(1 - np.abs(err).mean())


evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model.h5")

"""
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
"""

#evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model16.h5")


"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def evaluate_sig(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
=======
>>>>>>> b0216cf6b06aa0baadb47cc70d2e87f901352cab
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
        #sigmoid
        prob_whistle = pred_probs[:, 0]
        prob_not_whistle = 1 - prob_whistle
    else:
        prob_not_whistle = pred_probs[:, 0]
        prob_whistle = pred_probs[:, 1]

    pred_labels = (prob_whistle >= 0.95).astype(int)

    #save DataFrame
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

<<<<<<< HEAD
#evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model.h5")

"""
=======
evaluate_sig("D:\\Desk\\whistle_detection\\model\\sigmoid\\whistle_model.h5")
>>>>>>> b0216cf6b06aa0baadb47cc70d2e87f901352cab
