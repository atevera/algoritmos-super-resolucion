import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
import random
import h5py

class imset:
    """
        Clase encargada de generara el dataset a partir de una carpeta de imágenes sin 
        importar su cantidad. 

        Para ello, debe tener la imagen correspondiente a la variable path. 
        No hay restricción de tamaño ni cantidad de imágenes pero se recomienda 
        utilizar entre 10 a 15 imágenes dada la generalidad del algoritmo. 
    """

    def __init__(self, path, create = True, scale = 0.25, sigma = 1):
        self.path = path
        self.pathSource = path + '/Source'
        self.items_name = os.listdir(self.pathSource)
        self.pathFiles = [path+'/'+f for f in self.items_name]
        self.scale = scale
        self.sigma = sigma
        self.images = self.get_images(path+'/Source')
        if create:
            self.lowResolution, self.lowResolutionNormalized = self.make_low_resolution()
            self.interpolated, self.interpolatedNormalized = self.make_interpolation()
            self.HR_post = self.make_HR_postprocess()
        else:
            self.lowResolution = self.get_images(path+'/LR')
            self.lowResolutionNormalized = self.get_images(path+'/LRN')
            self.interpolated = self.get_images(path+'/IN')
            self.interpolatedNormalized = self.get_images(path+'/INN')
            self.HR_post = self.get_images(path+'/HR')

    def get_images(self, path):
        """
            Obtiene las imágenes dada un path como directorio. 
        """
        if not os.path.exists(path):
            print('Verifique que el directorio {} exista'.format(path))
            return []
        else: 
            items_name = os.listdir(path)
            pathFiles = [path+'/'+f for f in items_name]
            return [cv2.imread(image) for image in pathFiles]

    def make_low_resolution(self):
        """
            Función para reducir las dimensiones de una imagen a partir de un factor de escalado.
        """
        lowResolution = [cv2.resize(img, None, fx = self.scale, fy = self.scale) for img in self.images]
        lowResolutionNormalized = [self.postprocess(img) for img in lowResolution]
        self.save_dataset(lowResolution, self.path+'/LR')
        self.save_dataset(lowResolutionNormalized, self.path+'/LRN')
        return lowResolution, lowResolutionNormalized

    def make_interpolation(self):
        """
            Realiza la interpolación y normalización del conjunto de imágenes, así como 
            su almacenamiento en la ruta indicada. 
        """
        interpolated = [cv2.resize(img, None, fx = 1/self.scale, fy = 1/self.scale, interpolation = cv2.INTER_CUBIC)
                        for img in self.lowResolution]
        interpolatedNormalized = [self.postprocess(img) for img in interpolated]
        self.save_dataset(interpolated, self.path+'/IN')
        self.save_dataset(interpolatedNormalized, self.path+'/INN')
        return interpolated, interpolatedNormalized

    def make_HR_postprocess(self):
        """
            Ejecuta el postprocesado del conjunto de imágenes y posteriormnente realiza su almacenamiento en 
            la ruta indicada. 
        """
        HR_post = [self.postprocess(img) for img in self.images]
        self.save_dataset(HR_post, self.path+'/HR')
        return HR_post

    def postprocess(self,image):
        """
            Aplica el filtro pasa-altas a partir de la estrucutra de desenfoque Gaussiano, así
            como el normalizado de la imagen de entrada. 
        """
        img_filtered =  image - cv2.GaussianBlur(image, (0,0), self.sigma)
        img_norm = ((img_filtered - img_filtered.min()) * (1/(img_filtered.max() - img_filtered.min()) * 255)).astype('uint8')
        return img_norm

    def save_dataset(self, data, outpath):
        """
            Realiza el guardado del conjunto de imágenes dados y el directorio indicado. 
        """
        outpath_files = [outpath+'/'+name for name in self.items_name]
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        count = 0
        for img in data:
            cv2.imwrite(outpath_files[count],img)
            count += 1

class training_data:
    def __init__(self, imHRN, imLRN, sizes, name):
        self.imHRN = imHRN
        self.imLRN = imLRN
        self.name = name
        self.N = sizes[0]
        self.M = sizes[1]
        self.patchesHR, self.vectorsID = self.imgs_to_patches()
        
    def save_data(self):
        """
            Almacenamiento del conjunto de vectores en la base de datos con extensión .h5
        """
        f = h5py.File(self.name+'.h5', 'w')
        f.create_dataset(name = 'patches', data = self.patchesHR)
        f.create_dataset(name = 'vectors', data = self.vectorsID)
        f.close()

    def imgs_to_patches(self):
        """
            Permite seccionar al conjunto de imágenes
            en parches de acuerdo al tamaño indicado. 
        """
        n_images = len(self.imHRN)
        patchesHR = []
        vectorsID = []
        for i in range(n_images):
            print('Descomponiendo en parches imagen {}. Espere por favor... '.format(i+1))
            if i == 0:
                patchesHR, vectorsID = self.make_patches(i)
            else:
                tempHR, tempID = self.make_patches(i)
                patchesHR = np.row_stack((patchesHR, tempHR))
                vectorsID = np.row_stack((vectorsID, tempID))
        return patchesHR, vectorsID
 
    def make_patches(self, k):
        """
            Realiza el segmentado (mediante un barrido bidimensional) de la imagen alta y baja resolución en parches respectivamente.
            Posteriormente va concatenando los vectores de búsqueda y su correspondiente parche en alta calidad para construir la matriz 
            que será almacenada como base de datos. 
        """
        imgLR = self.imLRN[k]
        imgHR = self.imHRN[k]
        sz = imgHR.shape
        cnt = 0
        for i in range(2,sz[0]-1,4):
            for j in range(2,sz[1]-1,4):
                px = (i,j)
                cnt += 1
                patchHR = self.get_patch(imgHR, px, self.N)
                patchLR = self.get_patch(imgLR, px, self.M)
                
                tempLR = patchLR.flatten()
                tempHR = patchHR.flatten()

                f_row = patchHR[0,:,:]
                f_col = patchHR[:,0,:]
                f_row = f_row.flatten()
                f_col = f_col.flatten()
                tempSupp = np.concatenate((f_row, f_col))

                tempID = np.concatenate((tempLR, tempSupp))

                if cnt == 1:
                    dataID = tempID
                    dataHR = tempHR                
                else:
                    dataID = np.row_stack((dataID, tempID))
                    dataHR = np.row_stack((dataHR, tempHR))
        return dataHR, dataID

    def get_patch(self, image, center, size):
        """
            Permite 'cortar' el parche de la imagen dada respecto al centro o pixel de referencia. Genera un offset (borde negro) en caso el parche a obtener
            esté fuera de la imagen para no generar error. 
        """
        patch = np.zeros((size, size,3))
        offs = int((size - 1 )/2)
        img_border = self.make_borders(image, offs)
        patch = img_border[offs+center[0]-offs:offs+center[0]+offs+1, offs+center[1]-offs:offs+center[1]+offs+1, :]
        return patch
    
    def make_borders(self, image, offs):
        """
            Construye un borde en la iamgen para que no genere error al adquirir el parche en los pixeles de los laterales. 
        """
        image_border = cv2.copyMakeBorder(image, offs, offs, offs, offs, cv2.BORDER_CONSTANT, value = (0,0,0))
        return image_border


class superresolution:
    """
        Clase encargada de implementar el algoritmo propuesto por Freeman mediante el algoritmo de un paso.
        Dentro de los parámetros más importantes debe considerarse el factor de escalado, el factor de 
        superposición alpha y ya sea la imagen (parámetro image) o bien la ruta o 'path' donde se adquirirá. 

        Como parámetros extras tenemos el cambio del algoritmo de búsqueda, si los resultados se van a guardar y 
        en donde. 
    """
    def __init__(self, scale, alpha, name_dataset,  busqueda = 'ball_tree', save = False, pathOut = '', path = '', image = []):
        self.scale = scale
        self.name_dataset = name_dataset
        if path != '':
            self.path = path
            self.image = self.get_image()
        else:
            self.image = image
        self.image_input = self.preprocess()
        self.patchesHR, self.vectorsID = self.get_training_data()
        self.N = 5
        self.M = 7
        self.alpha = alpha
        self.scale_up = []
        self.busqueda = busqueda
        sz = self.image_input.shape
        offs = int((self.N - 1 )/2)
        self.high_frequencies = np.zeros((sz[0],sz[1],3))
        self.superresolution = self.algorithm(sz)
        if save:
            self.save_data(pathOut)


    def preprocess(self):
        """
            Preprocesado necesario para tener la imagen de entrada para el algoritmo de predicción.
        """
        kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
        image_filter = cv2.filter2D(self.image,-1,kernel)
        return cv2.resize(image_filter, None, fx = self.scale, fy = self.scale, interpolation = cv2.INTER_CUBIC)

    def get_training_data(self):
        """
            Carga la base de datos o diccionario donde están todos los parches para la predicción.
        """
        f = h5py.File(self.name_dataset + '.h5', 'r')
        patchesHR = f.get('patches')
        vectorsID = f.get('vectors')
        return patchesHR, vectorsID

    def get_image(self):
        """
            Devuelve la imagen solicitada
        """
        return cv2.imread(self.path)

    def meanAbs(self, image):
        """
            Calcula la media absoluta de la imagen dada y agrega un epsilon para evitar la indeterminación. 
        """
        return abs(np.mean(image)) + 0.0000000000000000000000001
                
    def predict_high_frequencies(self, sz):
        """
            Algoritmo de predicción de altas frecuencias a partir de algoritmos de búsqueda. 
        """
        cnt = 0
        # Carga del diccionario para la búsqueda del vector en el diccionario. 
        nbrs = NearestNeighbors(n_neighbors = 1, algorithm = self.busqueda).fit(self.vectorsID)
        # Adquisición de los parches para la construcción del vector de búsqueda. 
        for i in range(int(sz[0])):
            for j in range(int(sz[1])):
                px = (i,j)
                cnt += 1
                patchHF = self.get_patch(self.high_frequencies, px, self.N)
                patchLR = self.get_patch(self.image_input, px, self.M)
                
                mean = self.meanAbs(patchLR)
                
                tempLR = patchLR.flatten()

                f_row = patchHF[0,:,:]
                f_col = patchHF[:,0,:]

                f_row = f_row.flatten()
                f_col = f_col.flatten()

                tempSupp = np.concatenate((f_row, f_col)) * self.alpha

                vector_search = np.concatenate((tempLR, tempSupp))/mean

                _, index = nbrs.kneighbors([vector_search])
                index = index[0]

                self.put_pixel(px, index, mean)

    def put_pixel(self, center, index, mean):
        """
            Asigna el pixel en la imagen de acuerdo al algoritmo de predicción para mejorar la resolución de la imagen. 
        """
        patchHR = self.patchesHR[index]
        patchHR = np.reshape(patchHR, (5, 5, 3))
        patchHR = patchHR*mean
        self.high_frequencies[center[0],center[1], :] = patchHR[2,2,:]

    def get_patch(self, image, center, size):
        """
            Adquiere el parche de baja resolución para su mejora. 
        """
        patch = np.zeros((size, size,3))
        offs = int((size - 1 )/2)
        img_border = self.make_borders(image, offs)
        patch = img_border[offs+center[0]-offs:offs+center[0]+offs+1, offs+center[1]-offs:offs+center[1]+offs+1, :]
        return patch

    def algorithm(self, sz):
        """
            Algoritmo de Freeman dada la suma de la imagen escalada y las altas frecuencias predecidas. 
        """
        self.scale_up = cv2.resize(self.image, None, fx = self.scale, fy = self.scale, interpolation = cv2.INTER_CUBIC)
        self.predict_high_frequenciesV2(sz)
        return self.high_frequencies + self.scale_up

    def make_borders(self, image, offs):
        """
            Construye un borde en la iamgen para que no genere error al adquirir el parche en los pixeles de los laterales. 
        """
        image_border = cv2.copyMakeBorder(image, offs, offs, offs, offs, cv2.BORDER_CONSTANT, value = (0,0,0))
        return image_border

    def save_data(self, pathOut):
        """
            Almacena los resultados de aplicar el algoritmo de súper resolución en caso se solicite.
        """
        path, name = pathOut
        img_sr = path + '/' + name + 'sr_a' + str(self.alpha) + '_e' + str(self.scale) + '.jpg'
        img_hr = path + '/' + name + 'hr_a' + str(self.alpha) + '_e' + str(self.scale) + '.jpg'
        cv2.imwrite(img_sr, self.superresolution)
        cv2.imwrite(img_hr, self.high_frequencies)

    def get_superresolution(self):
        """
            Devuelve la imagen en mejor resolución que la entrada.
        """
        return self.superresolution

    def get_high_frequencies(self):
        """
            Devuelve la imagen de altas frecuencias dado el algoritmo de predicción. 
        """
        return self.high_frequencies