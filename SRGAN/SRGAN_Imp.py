
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


#Importamos el modelo generador previamente entrenado
from keras.models import load_model

generator = load_model('gen_e_.h5', compile=False)


#ingresamos nuestra imagen de baja resolución y procedemos a recortar un area de 32x32 o el numero
#de pixeles cque necesite de netrada nuestro modelo, en nuestro caso es de 32x32
crop_lr=cv2.imread("Saltillo.png")
sq = 32
x = 466
y = 343
data_lr = crop_lr[y:y+sq, x:x+sq, :]
data_hr = cv2.imread("Saltillo.png")

 
#Cambio de BGR a RGB para mostrar. 

data_lr = cv2.cvtColor(data_lr, cv2.COLOR_BGR2RGB)
data_hr = cv2.cvtColor(data_hr, cv2.COLOR_BGR2RGB)
img_rec = cv2.rectangle(data_hr, (x,y), (x+sq,y+sq), [0,255,0], 3)

data_lr = data_lr / 255.
data_hr = data_hr / 255.

data_lr = np.expand_dims(data_lr, axis=0)
data_hr = np.expand_dims(data_hr, axis=0)

generated_data_hr = generator.predict(data_lr) #Le indicamos al generador los datos que debe predecir




# Muestra las 3 imagenes
plt.figure(figsize=(8, 12))
plt.subplot(231)
plt.title('Imagen LR')
plt.imshow(data_lr[0,:,:,:])
plt.subplot(232)
plt.title('Superresolución')
plt.imshow(generated_data_hr[0,:,:,:])
plt.subplot(233)
plt.title('Imagen Original HR')
plt.imshow(img_rec)

plt.show()

##########################################################################################
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
fig.tight_layout()

plt.title('Imagen LR')
ax[0].imshow(data_lr[0,:,:,:])              #Comparativa mas a detalle de las imagenes

plt.title('Superresolución')
ax[1].imshow(generated_data_hr[0,:,:,:])



