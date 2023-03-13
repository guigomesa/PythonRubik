import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definindo parâmetros
input_shape = (224, 224, 3)
num_classes = 1
batch_size = 16
epochs = 10

# Definindo caminhos dos diretórios de treinamento e validação
train_dir = 'cubo_treinamento'
validation_dir = 'cubo_validacao'

# Criando geradores de dados para treinamento e validação
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'cubo_treinamento',
    classes=['cubo'],
    class_mode='categorical',
    target_size=(224, 224),
    batch_size=batch_size
)

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    'cubo_validacao',
    classes=['cubo'],
    class_mode='categorical',
    target_size=(224, 224),
    batch_size=batch_size
)

# Criando modelo
model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
model.trainable = False

# Adicionando camadas ao modelo
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

# Compilando modelo
model = tf.keras.models.Model(model.input, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando modelo
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Salvando modelo treinado
model.save('modelo_cubo.h5')