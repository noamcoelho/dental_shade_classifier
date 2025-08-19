import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from glob import glob

MODEL_PATH = 'dental_shade_classifier_model.h5'
TEST_DIR = os.path.join('dental_shade_classifier', 'datasets', 'test')
IMG_HEIGHT, IMG_WIDTH = 128, 128

model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted(os.listdir(os.path.join('dental_shade_classifier', 'datasets', 'train')))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

total = 0
correct = 0
confidences = []

for class_name in class_names:
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_path in glob(os.path.join(class_dir, '*.jpeg')):
        total += 1
        pred_class, conf = predict_image(img_path)
        confidences.append(conf)
        real_class = class_name
        print(f'Imagem: {os.path.basename(img_path)} | Real: {real_class} | Prevista: {pred_class} | Confiança: {conf:.2f}')
        if pred_class == real_class:
            correct += 1

if total > 0:
    acc = correct / total
    mean_conf = np.mean(confidences)
    print(f'\nAcurácia: {acc:.2%}')
    print(f'Confiança média: {mean_conf:.2f}')
else:
    print('Nenhuma imagem encontrada para avaliação.')
