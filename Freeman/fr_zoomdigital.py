import example_based_super_resolution as sr
import cv2
import matplotlib.pyplot as plt

# Imagen propuesta para súper resolución
path = 'implementacion/Saltillo.png'
img1 = cv2.imread(path)
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Coordenadas/área de interés
sq = 32
x = 464
y = 341
cut = img[y:y+sq, x:x+sq, :]

# Factor de superposición
alpha = 0.1*((7^2)/(2*5-1))

# Ejecución de algoritmo
img_sr = sr.superresolution(4, alpha, 'diccionario', image = cut)
# Adquisición de imagen en alta resolución
sr_ = img_sr.get_superresolution()
# Normalizado
new_sr = ((sr_ - sr_.min()) * (1/(sr_.max() - sr_.min()) * 255)).astype('uint8')


# Presentación de resultados
fig, ax = plt.subplots(1,4)
ax[1].imshow(cut)
ax[2].imshow(new_sr)
img_rec = cv2.rectangle(img, (x,y), (x+sq,y+sq), [0,255,0], 3)
ax[0].imshow(img_rec)
sr_ = img_sr.get_high_frequencies()
new_sr = ((sr_ - sr_.min()) * (1/(sr_.max() - sr_.min()) * 255)).astype('uint8')
ax[3].imshow(new_sr)
plt.show()