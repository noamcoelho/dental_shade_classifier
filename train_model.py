import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import os

# Caminhos dos diretórios de treino e teste
train_dir = os.path.join('dental_shade_classifier', 'datasets', 'train')
test_dir = os.path.join('dental_shade_classifier', 'datasets', 'test')

# Parâmetros
img_height, img_width = 128, 128
batch_size = 16

# Pré-processamento e aumento de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)


# Transfer Learning com ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Congela as camadas convolucionais
# (camadas convolucionais são aquelas que extraem características da imagem)

model = models.Sequential([
    # o sequential serve para empilhar várias camadas (o nome deixa claro que a saída de uma camada é a entrada da próxima)
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento
history = model.fit(
    # o comando fit faz o treinamento do modelo usando os dados de treinamento
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Avaliação
loss, acc = model.evaluate(test_generator)
# o comando evaluate calcula a perda e a acurácia do modelo usando os dados de teste
print(f'Acurácia no teste: {acc:.2f}')

# Salvar modelo
model.save('dental_shade_classifier_model.keras')
print('Modelo salvo como dental_shade_classifier_model.keras')
