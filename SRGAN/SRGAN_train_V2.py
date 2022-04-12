# Importar Librerias
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn.model_selection import train_test_split
from keras import layers, Model                                                 
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten          #la libreria keras se encaraga de cargar lo basico
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add         #como las funciones de activacion, el escalado, normalización, etc.
from keras.models import Sequential                                         
from keras.models import load_model
from keras.applications.vgg19 import VGG19
#*Nota sobre VGG19.
#VGG19 is used for the feature map obtained by the j-th convolution (after activation) 
#before the i-th maxpooling layer within the VGG19 network. 
# VGG architecture: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
from tqdm import tqdm
#--------------------------------------------------------------
#Definir bloques residuales para el entrenamiento del generador, estos ayudan a evitar el problema del descenso de gradiente.

def res_block(ip):
    
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    
    return add([ip,res_model])

#Bloques de escalado
def upscale_block(ip):
    
    up_model = Conv2D(256, (3,3), padding="same")(ip)

#Se utiliza la función UpSampling2D para el escalado de la imagen
# y aplica una interpolación de vecinos cercanos.
    up_model = UpSampling2D( size = 2 )(up_model) 
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model
#--------------------------------------------------------------
#Función del modelo generador.
def create_gen(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)
#--------------------------------------------------------------
#Creamos el bloque del discriminador

def discriminator_block(ip, filters, strides=1, bn=True):
    
    disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)
    
    if bn:
        disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    
    return disc_model


#Discriminador, tomando en cuenta la estructura del paper
def create_disc(disc_ip):

    df = 64
    
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)
#--------------------------------------------------------------
#VGG19 (Visual Geometry Group) nuestra red convolucional dedicada a la clasificación de imagenes 
#consta de 16 capas convolucionales, 3 capas completamente conectadas, 5 capas de 
# Maxpooling(Valores máximos de los parches)  y 1 capa de SoftMax.

#Construye un modelo VGG19 pre-entrenado con imagenet(base de datos) cuya salida son las 
#caracteristicas de la imagen extraidas en el tercer bloque de el modelo


def build_vgg(hr_shape):
    
    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
    
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

#Combinación del modelo:

def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip) #El generador realiza una imagen falsa
    
    gen_features = vgg(gen_img)
    
    disc_model.trainable = False
    validity = disc_model(gen_img) #La imagen falsa es la entrada del discriminador
    
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features]) #Con la validación
    #minimizamos las perdidas 

# Se obtienen 2 perdidas, adversaria y de contenido (VGG loss)
#Adversaria: la cual se basa en las probabilidades del discriminador de dscernir correctamente sobre todos las muestras de entrenamiento.
# y utiliza la función binary_crossentropy

#Contenido: mapeo de caracteristicas obtenida por la j-esima convolución (despues de la activación) 
#antes de la  i-esima capa de maxpoolación (maxpooling) dentro de la red VGG19.
# MSE entre las representaciones caracteristicas de la imagen generada y la imagen real. 

#--------------------------------------------------------------
n=13 # Número de imágenes en el dataset (Toma las primeras n imagenes)

lr_list = os.listdir("data/lr_images")[:n]

#Obtenemos las imagenes de baja y alta resolución mediante un ciclo iterativo
#desde las carpetas donde se encuentren ambos datasets.
lr_images = []
for img in lr_list:
    img_lr = cv2.imread("data/lr_images/" + img)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)   


hr_list = os.listdir("data/hr_images")[:n]
   
hr_images = []
for img in hr_list:
    img_hr = cv2.imread("data/hr_images/" + img)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)   

lr_images = np.array(lr_images) #Se agregan a un array
hr_images = np.array(hr_images)

import numpy as np

#Escalado de valores.
lr_images = lr_images / 255.
hr_images = hr_images / 255.

#División de arrays o matrices en subsets de entrenamiento y test
#(Sklearn.model Función train_test_split)
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, 
                                                      test_size=0.33, random_state=42)

hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])  #Obtener la forma deseada
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])  #de entrada de las imagenes
                                                                      #al generador y discriminador
lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

generator = create_gen(lr_ip, num_res_block = 16) #Crea un modelo generador para implementar
#generator = load_model('gen_e_500.h5', compile=False) 
#Carga una configuración de modelo para generar un modelo mas precíso
generator.summary()

discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
discriminator.summary()
#binary_crossentropy nos ayuda a obtener grandes perdidas para valores erroneos
#y valores pequeños de perdida para cuando los valores tienden a ser correctos
# se utiliza el optimizador adam y las metricas de accuracy nos permiten verificar
#si el discriminador logro una buena diferenciación o se le penaliza por obtener un 
#dato erroneo.
#--------------------------------------------------------------
vgg = build_vgg((128,128,3))
print(vgg.summary())
vgg.trainable = False #Extraer caracteristicas del modelo VGG para combinarlo con el generador
                      # y crear el modelo GAN                      
gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
#--------------------------------------------------------------
gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
gan_model.summary()
#--------------------------------------------------------------
#Creamos la lista de imagenes de Baja y Alta Resolución en batches que seran obtenidas
#del batch durante durante el entrenamiento
 
batch_size = 1  
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])
#--------------------------------------------------------------  
#Epocas del entrenamiento
epochs = 5  #Se asigna un numero deseado de epocas para el entrenamiento

g_value = []
d1_value = []
d2_value = []
#Enumeramos el entrenamiento por epocas
for e in range(epochs):
    
    fake_label = np.zeros((batch_size, 1)) # Asigna una etiqueta de 0 a las imagenes generadas (fake).
    real_label = np.ones((batch_size,1)) # Asigna una etiqueta de 1 a las imagenes reales.
    
    #Creamos listas vacias para los datos de perdida del generador y el discrimiador. 
    g_losses = []
    d_losses = []
    
    #Enumerar el entrenamiento por batches. 
    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b] #Obtiene un batch de imagenes de LR para entrenar.
        hr_imgs = train_hr_batches[b] #Obtiene un batch de imagenes de HR para entrenar.
        
        fake_imgs = generator.predict_on_batch(lr_imgs) #Definimos las imagenes falsas
        #que se predicen dependiendo de los batches
        
        #Entrenamiento del discriminador con las imagenes falsas y las reales. 
        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
        
        #Entrenamos el generador desactivando el entrenamiento para el discriminador
        discriminator.trainable = False
        
        #Promedio de las perdidas del discriminador para resultados(perdida de la imagen real y la falsa). 
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real) 
        
        #Extraemos las caracteristicas desde VGG, para el calculo de la perdida de contenido
        image_features = vgg.predict(hr_imgs)
     
        #Entrenamiento por GAN. 
        #Entrenamiento de una imagen a 1 solo batch
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
        
        #Guardamos las perdidas en las listas . 
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
    #Perdidas se convierten en arrays para mejor manejo de los datos al promediar.   
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    
     #Cálculo del promedio de las perdidas del generador y discriminador
    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)

    g_value.append(g_loss)
    d1_value.append(d_loss[0])  #Variables para obtener graficas de perdidas y precisión.
    d2_value.append(d_loss[1])
    
    #Visualizamos las epocas y las perdidas durante el entrenamiento. 
    print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)

#-------------------------------------------------------------- 
#Guarda el generador despues del entrenamiento dependiendo del numero de epocas
generator.save("gen_e_"+ str(e+1) +".h5")
#--------------------------------------------------------------
#Graficamos las perdidas del generador asi como las del discriminador, tomamos en cuenta
#la precisión del discriminador todo esto respecto a las epocas de entrenamiento

epocas = np.array([x for x in range(1,epochs+1)])
fig, ax = plt.subplots(1,3)
ax[0].plot(epocas, g_value)
ax[1].plot(epocas, d1_value)
ax[2].plot(epocas, d2_value)

ax[0].set_title("Pérdidas del generador")
ax[1].set_title("Pérdidas del discriminador")
ax[2].set_title("Precisión del discriminador")

ax[0].set_xlabel("Épocas")
ax[1].set_xlabel("Épocas")
ax[2].set_xlabel("Épocas")

#--------------------------------------------------------------
#Valores máximos y mínimos de las perdidas
print('Pérdida generador -- max: {}, min: {}'.format(np.max(g_value), np.min(g_value)))
print('Pérdida discriminador -- max: {}, min: {}'.format(np.max(d1_value), np.min(d1_value)))
print('Precisión discriminador -- max: {}, min: {}'.format(np.max(d2_value), np.min(d2_value)))
plt.show()