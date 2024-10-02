import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Ruta a las carpetas de imágenes
data_dir = 'database/Brocoli'

# Parámetros
img_size = (100, 100)
batch_size = 32
num_classes = 2  # Dos clases: bueno y malo

# Cargar imágenes desde las carpetas usando ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Escalar las imágenes y dividir en entrenamiento/validación

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Conjunto de entrenamiento
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Conjunto de validación
)

# Definir un bloque residual
def residual_block(x, filters, kernel_size=3, stride=1, activation='relu'):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation(activation)(x)
    
    return x

# Construir y entrenar el modelo ResNet
def build_resnet(input_shape, num_classes, filters, kernel_size=3, stride=1):
    inputs = layers.Input(shape=input_shape)
    
    # Primera capa convolucional
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    
    # Añadir bloques residuales
    x = residual_block(x, filters=filters)
    x = residual_block(x, filters=filters*2, stride=stride)
    x = residual_block(x, filters=filters*4, stride=stride)
    
    # Global Average Pooling y capas densas
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Capa de salida
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Crear el modelo
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Configuración de parámetros
filter_options = [32, 64, 128]
kernel_size_options = [3, 5]
stride_options = [1, 2]

for filters in filter_options:
    for kernel_size in kernel_size_options:
        for stride in stride_options:
            # Crear el modelo ResNet
            model = build_resnet(input_shape=(100, 100, 3), num_classes=num_classes, filters=filters, kernel_size=kernel_size, stride=stride)

            # Configurar TensorBoard
            log_dir = f"./resnet/logs/filters_{filters}_kernel_{kernel_size}_stride_{stride}/"
            model_dir = f"./resnet/model/filters_{filters}_kernel_{kernel_size}_stride_{stride}/"
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Entrenar el modelo con el callback de TensorBoard
            model.fit(
                train_generator,
                epochs=14,
                validation_data=validation_generator,
                callbacks=[tensorboard_callback]
            )

            # Guardar el modelo
            model.save(f'./resnet/model/modelo_resnet_filters_{filters}_kernel_{kernel_size}_stride_{stride}.keras')

            # Instrucciones para visualizar TensorBoard
            print(f"Ejecuta el siguiente comando en tu terminal para visualizar TensorBoard:")
            print(f"tensorboard --logdir={log_dir}")
