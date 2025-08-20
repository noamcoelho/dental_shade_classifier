import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODEL_PATH = 'dental_shade_classifier_model.keras'
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
    return predicted_class

y_true = []
y_pred = []

for class_name in class_names:
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_path in glob(os.path.join(class_dir, '*.jpeg')):
        y_true.append(class_name)
        pred_class = predict_image(img_path)
        y_pred.append(pred_class)

if y_true:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.show()
else:
    print('Nenhuma imagem encontrada para avaliação.')
