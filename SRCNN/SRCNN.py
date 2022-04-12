import sys
import keras
import cv2
import numpy
import matplotlib
import skimage
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.experimental.numpy as tnp

# import the necessary packages
from collections import namedtuple
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import adam_v2
from matplotlib import pyplot as plt
from IPython.display import clear_output
import keras.preprocessing.image_dataset as kds
import cv2
import numpy as np

class SR:
    ColSpace = namedtuple('ColorSpace','RGB YCrCb')
    ColorSpace = ColSpace(RGB = 0, YCrCb = 1)
    #self.RGB = 0
    #self.YCrCb = 1
    
    def model(self,ColorSpaceSelected):
        SRCNN = Sequential()
        
        if ColorSpaceSelected == self.ColorSpace.RGB:
            # Agregar las capas del modelo, en orden secuencial
            SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True, strides = 1, input_shape=(None, None, 3)))
            SRCNN.add(Conv2D(filters=64, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True))
            SRCNN.add(Conv2D(filters=3, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='linear', padding='same', use_bias=True))
        
        if ColorSpaceSelected == self.ColorSpace.YCrCb:
            # Agregar las capas del modelo, en orden secuencial
            SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True, strides = 1, input_shape=(None, None, 1)))
            SRCNN.add(Conv2D(filters=64, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True))
            SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='linear', padding='same', use_bias=True))
        
        # Definir optimizador del modelo, para este caso utilizaremos el optimizados Adam, para capas 1 y 2 lr=3e-3
        # y para la capa 3 lr=3e-4
        # adam1 = adam_v2.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
        # adam2 = adam_v2.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
        adam = adam_v2.Adam(learning_rate=0.003)
        #optimizers_and_layers = [(adam1, SRCNN.layers[:1]), (adam2, SRCNN.layers[2])]
        #adam = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        
        # Compilar el modelo
        SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
        
        return SRCNN

    def getImage(self,LRImage,Colorspace,TrainedModelPath = None):
        SRCNN = Sequential()
        
        if Colorspace == 0:                 # Espacio de color RGB
            # Agregar las capas del modelo, en orden secuencial
            SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True, strides = 1, input_shape=(None, None, 3)))
            SRCNN.add(Conv2D(filters=64, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True))
            SRCNN.add(Conv2D(filters=3, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='linear', padding='same', use_bias=True))
        
        if Colorspace == 1:                 # Espacio de YCrCb:
            # Agregar las capas del modelo, en orden secuencial
            SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True, strides = 1, input_shape=(None, None, 1)))
            SRCNN.add(Conv2D(filters=64, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='relu', padding='same', use_bias=True))
            SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                             activation='linear', padding='same', use_bias=True))
        
        # Definir optimizador del modelo, para este caso utilizaremos el optimizados Adam, para capas 1 y 2 lr=3e-3
        # y para la capa 3 lr=3e-4
        #adam1 = adam_v2.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
        #adam2 = adam_v2.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
        #optimizers_and_layers = [(adam1, SRCNN.layers[:1]), (adam2, SRCNN.layers[2])]
        #adam = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        adam = adam_v2.Adam(learning_rate=0.003)
        
        # Compilar el modelo
        SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
        
        #SRCNN = self.model(self,ColorSpace)
        
        tmpImg = np.zeros(LRImage.shape)                                         # Initialize variable
        LRImage = cv2.cvtColor(LRImage, cv2.COLOR_BGR2RGB)                       # BGR Image assumed
        
        if TrainedModelPath != None:
            SRCNN.load_weights(TrainedModelPath)
        
        if Colorspace == 0:                    # Espacio de color RGB
            tmpImg = LRImage.astype(np.float32)/255
            dim = tmpImg.shape

            # Preparar entrada para predicción por la red
            Input_train = np.zeros((1,dim[0],dim[1],dim[2]))
            Input_train[0,:,:,:] = tmpImg[:,:,:]
            
        elif Colorspace == 1:                 # Espacio de color YCrCb
            tmpImg = cv2.cvtColor(LRImage, cv2.COLOR_RGB2YCrCb).astype(np.float32)/255
            dim = tmpImg.shape

            # Preparar entrada para predicción por la red
            Input_train = np.zeros((1,dim[0],dim[1],1))
            Input_train[0,:,:,0] = tmpImg[:,:,0]

        # Predecir el resultado de image
        salida_test = SRCNN.predict(Input_train, batch_size = 1)

        if Colorspace == 0:                # Espacio de color RGB:
            # Pos-procesamiento del resultado de salida de la red
            tmp = salida_test[0,:,:,:]

            for index,x in np.ndenumerate(tmp):
                if tmp[index] < 0:
                    tmp[index] = 0
                if tmp[index] > 1:
                    tmp[index] = 1

            # Dar formato a imagen RGB con valores de [0,255]
            tmp = (tmp*255).astype(np.uint8)
            return cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)                  # Return an BGR image
        
        elif Colorspace == 1:              # Espacio de color YCrCb:
            # Pos-procesamiento del resultado de salida de la red
            tmp = np.copy(tmpImg)
            tmp[:,:,0] = salida_test[0,:,:,0]

            for index,x in np.ndenumerate(tmp[:,:,0]):
                index = index + tuple([0])
                if tmp[index] < 0:
                    tmp[index] = 0
                if tmp[index] > 1:
                    tmp[index] = 1
            
            # Convertir del espacio de color YCrCb a RGB
            tmp = cv2.cvtColor((tmp*255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
            return tmp