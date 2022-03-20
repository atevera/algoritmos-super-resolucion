import cv2
import os
import argparse
import matplotlib.pyplot as plt
import prepare_dataset as pds
import numpy as np

def normalized(image):
    mean = abs(np.mean(image)) + 0.0000000000000000000000001
    return image/mean

def filter_pass(image):
    kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
    #filter the source image
    return cv2.filter2D(image,-1,kernel)


if __name__ == "__main__":

    pathHR = 'dataset/HR'
    pathLR = 'dataset/LR'
    files = pds.get_files(pathHR)
    pathFilesHR = pds.make_directory(pathHR, files)
    pathFilesLR = pds.make_directory(pathLR, files)
    imagesHR = pds.get_images(pathFilesHR)
    imagesLR = pds.get_images(pathFilesLR)
    n = 5
    img_RGB = cv2.cvtColor(imagesLR[n], cv2.COLOR_BGR2RGB)
    img_filter = filter_pass(imagesLR[n])
    img_norm = normalized(img_filter)
    x = 500
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img_RGB)
    ax[1].imshow(img_filter)
    ax[2].imshow(img_norm)
    ax[0].set_title('Original')
    ax[1].set_title('Filtro Pasa Altas')
    ax[2].set_title('Normalizada')
    plt.show()