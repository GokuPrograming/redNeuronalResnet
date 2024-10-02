import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Ruta al modelo y tamaño de imagen
ruta_modelo = './resnet/model/modelo_resnet_filters_32_kernel_3_stride_2.keras'
img_size = (100, 100)

def predecir_imagen(model, ruta_imagen):
    # Cargar la imagen
    img = image.load_img(ruta_imagen, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0  # Escalar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir batch dimension
    
    # Hacer la predicción
    prediction = model.predict(img_array)
    return prediction

def analizar_carpeta(carpeta_imagenes, ruta_modelo):
    # Cargar el modelo
    model = tf.keras.models.load_model(ruta_modelo)
    
    # Iterar sobre las imágenes en la carpeta
    resultados = {}
    for nombre_archivo in os.listdir(carpeta_imagenes):
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
        if os.path.isfile(ruta_imagen):
            prediccion = predecir_imagen(model, ruta_imagen)
            clase = ['Bueno', 'Malo'][np.argmax(prediccion)]
            resultados[nombre_archivo] = clase
            print(f'Imagen: {nombre_archivo}, Predicción: {clase}')
    
    return resultados

# Ruta a la carpeta de imágenes
carpeta_imagenes = './resnet/src/'

# Analizar todas las imágenes en la carpeta
resultados = analizar_carpeta(carpeta_imagenes, ruta_modelo)
print("Resultados:", resultados)
