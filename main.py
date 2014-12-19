import sys
from utilities import FileHelper
import constants

''' Import de los algoritmos '''
import LloydsAlgorithm.VectorialCuantification
import BayesAlgorithm.Bayes

sys.path.append( "SOM-Algorithm" )
import SOM

sys.path.append( "k-medidas/" )
from KMeans import KMeans,Class as KMeansClass

def loadFileKMeans(file,classNameIndex):
	
	k = KMeans(constants.getN())
	fileHelper = FileHelper()
	

	c1 = KMeansClass(0,"setosa")
	c1.setVCenter([4.6,3.0,4.0,0.0])
	c2 = KMeansClass(1,"versicolor")
	c2.setVCenter([6.8,3.4,4.6,0.7])

	k.addClass(c1)
	k.addClass(c2)
	try:
		f = fileHelper.openReadOnlyFile(file)
		
		lineas = f.readlines()
		uMatrix = constants.getKMeansInitializeUMAtrix(len(lineas))
		k.setUMatrix(uMatrix)
		xVector = []

		for linea in lineas:

			xVector = linea.strip("\r\n").split(",")
			del xVector[classNameIndex-1]
			xVector = [float(x) for x in xVector]
			print xVector

			k.addXVector(xVector)

		return k
	except:
		print("Error al leer el fichero")


if __name__ == "__main__":
	

	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "+++++++++++"
	print "+++++++++++                KMEANS"
	print "+++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"

	k = loadFileKMeans("Iris2Clases.txt",5)

	print ">>> Carga finalizada"

	k.doTraining(epsilonLimit=constants.getKMeansEpsilon(),b=constants.getKMeansB())
