import example_based_super_resolution as sr

# Construye el dataset a partir de la carpeta indicada y dentro de ella deberán estar las 
# imágenes en una carpeta de nombre Source. 
# También puede ser que cargue las imágenes únicamente si el parámetro se define: create = False
dset = sr.imset('dataset', create = False)

# Obtiene las imágenes en alta y baja resolución normalizadas y filtradas para 
# su almacenamiento en el diccionario. 
imLRN = dset.interpolatedNormalized
imHRN = dset.HR_post

# Construye los parches correspondientes y los almacena en el archivo con nombre 
# euivalente al parámetro de tipo String. La tupla indica el tamaño de los parches 
# en baja y alta resolución. 
train = sr.training_data(imHRN, imLRN, (5,7), 'dataset_sgm1_new')

# Reporte de finalización de proceso
print('Tamaño dataset ParchesHR', train.patchesHR.shape)
print('Tamaño dataset VectorsID', train.vectorsID.shape)
print('Guardando información, espere ...')

# Guardado de datos en archivo h5
train.save_data()
