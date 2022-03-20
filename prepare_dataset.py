import cv2
import os
import argparse
import matplotlib.pyplot as plt

def get_args():
    """
    Lee los argumentos de consola para la configuración de la base de datos. 
    Es muy importante cumplir con ambos parámetros para que todo funcione adecuadamente. 

    Al correr el script, considere como ejemplo:
    >> python3 .\prepare_dataset.py --dim 500 --reduce 0.25
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', dest='dim_img', type = int, help='Add dimension to check')
    parser.add_argument('--reduce', dest='reduce_img', type = float, help='Add reduce factor')
    args = parser.parse_args()
    return args.dim_img, args.reduce_img

def get_files(path):
    """
    Adquiere los nombre de todas las imágenes en el directorio indicado.
    """
    return os.listdir(path)

def make_directory(path, files):
    """
    Construye la lista de rutas para cada una de las imágenes a utilizar. 
    """
    return [path+'/'+f for f in files]

def get_images(path):
    """
    Almancena en una lista todas las imágenes del directorio. 
    """
    return [cv2.imread(image) for image in path]

def check_size(dataset, files, size):
    """
    Verifica que todas las imágenes sean cuadradas de dimensión 'size'.
    """
    count = 0
    for img in dataset:
        alto, ancho, _ = img.shape
        if not ( alto == size and ancho == size):
            print('Verifique las dimensiones de la imagen', files[count])
            return 0
        count += 1
    print('Todas las imagenes cumplen con la especificaciones adecuadas')

def make_low_resolution(dataset, outpath, scale):
    """
    Función para reducir las dimensiones de una imagen a partir de un factor de escalado.
    """
    count = 0
    for img in dataset:
        img_LR = cv2.resize(img, None, fx = scale, fy = scale)
        cv2.imwrite(outpath[count],img_LR)
        count += 1

def apply_bicubic(dataset, outpath, scale):
    """
    Aplica interpolación bicúbica para aumentar la cantidad de pixeles dada una imágen de baja resolución. 
    """
    count = 0
    for img in dataset:
        img_CS = cv2.resize(img, None, fx = 1/scale, fy = 1/scale, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(outpath[count],img_CS)
        count += 1


if __name__ == "__main__":
    dim_img, reduce_img = get_args()

    pathHR = 'dataset/HR'
    pathLR = 'dataset/LR'
    pathCS = 'dataset/CS'

    files = get_files(pathHR)
    pathFilesHR = make_directory(pathHR, files)
    pathFilesLR = make_directory(pathLR, files)
    pathFilesCS = make_directory(pathCS, files)
    imagesHR = get_images(pathFilesHR)
    check_size(imagesHR, files, dim_img)
    make_low_resolution(imagesHR, pathFilesLR, reduce_img)
    imagesLR = get_images(pathFilesLR)
    apply_bicubic(imagesLR, pathFilesCS, reduce_img)