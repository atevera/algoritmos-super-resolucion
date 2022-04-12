Para correr el algoritmo, se recomienda crear un ambiente en miniconda e instalarlo
a partir del archivo "requirements.txt" siguiendo el comando:
$ conda create --name sr_freeman --file requirements.txt

En esta carpeta se encuentran los siguientes archivos:
	- diccionario.h5 --> Contiene el diccionario donde se realizará la búsqueda de parches.

	- example_based_super_resolution.py 
		Librería desarrollada por nosotros donde se encuentran
		todas las herramientas necesarias para crear, implementar
		y crear el algoritmo propuesto por Freeman. 

	- fr_make_dataset_dic.py
		Archivo para generar y almacenar el dataset y/o diccionario
		a partir de un conjunto de imágenes que debe estar en el siguiente
		directorio: ./dataset/Source

	- fr_zoomdigital.py
		Ejemplo de aplicación del algoritmo dada una imagen y un área
		de interés. Como ejemplo se han compartido tres imágenes cuyas
		áreas de interés son las siguientes. 

	- requirements.txt
		Lista de paquetes necesarios para utilizar el algoritmo. 
		