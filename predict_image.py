import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Caminho do modelo salvo
MODEL_PATH = 'dental_shade_classifier_model.h5'
# Caminho da imagem para teste
IMAGE_PATH = 'C://Users//coelh//Downloads//dental_shade_classifier//dental_shade_classifier//datasets//train//A1//01.jpeg'
# Parâmetros de entrada do modelo
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Carregar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Carregar classes
train_dir = os.path.join('dental_shade_classifier', 'datasets', 'train')
class_names = sorted(os.listdir(train_dir))

# Função para prever a classe de uma imagem
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

if __name__ == '__main__':
    predicted_class, confidence = predict_image(IMAGE_PATH)
    print(f'Classe prevista: {predicted_class} (confiança: {confidence:.2f})')