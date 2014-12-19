import sys
from utilities import FileHelper
import constants

''' Import de los algoritmos '''
import LloydsAlgorithm.VectorialCuantification
import BayesAlgorithm.Bayes

sys.path.append( "SOM-Algorithm" )
import SOM

sys.path.append( "k-medidas/" )
from KMeans import *


def readMedidas(file):
	fileHelper = FileHelper()

	try:
		f = fileHelper.openReadOnlyFile(file)

		lineas = f.readlines()
		medidas = []
		for linea in lineas:
			linea = linea.split(",")
			linea[len(linea)-1] = linea[len(linea)-1].strip("\r\n")

			if ( linea != "\r\n" or linea != "\n" ):
				for m in linea:
					muestra = m.split(":")
					medidas.append((muestra[0], muestra[1]))
	except:
		print("Error al leer el fichero")

	return medidas

def loadFileKMeans(file):
	

	fileHelper = FileHelper()

	try:
		f = fileHelper.openReadOnlyFile(file)

		lineas = f.readlines()
		xVector = []
		for linea in lineas:
			xVector = linea.strip("\r\n").split(",")

	except:
		print("Error al leer el fichero")

if __name__ == "__main__":
	
	loadFileKMeans("Iris2Clases.txt")