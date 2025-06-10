import tensorflow as tf

model = tf.keras.models.load_model('model/whistle_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic Range Quantization
tflite_model = converter.convert()

with open('model/whistle_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite模型大小: {len(tflite_model) / 1024:.2f} KB")
