import sys
from utilities import FileHelper
import constants

''' Algorithm imports '''
import LloydsAlgorithm.VectorialCuantification
from BayesAlgorithm.Bayes import Bayes, Class as BayesClass

sys.path.append( "SOM-Algorithm" )
import SOM

sys.path.append( "k-medidas/" )
from KMeans import KMeans,Class as KMeansClass
''' End of algorithm imports '''

''' This method loads KMeans needed values and adds the to the wrapper class KMeans '''

def loadFileKMeans(file,classNameIndex):
	
	k = KMeans(constants.getN())
	fileHelper = FileHelper()
	

	c1 = KMeansClass(0,"Iris-setosa")
	c1.setVCenter([4.6,3.0,4.0,0.0])
	c2 = KMeansClass(1,"Iris-versicolor")
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

			k.addXVector(xVector)

		return k
	except:
		print("Error al leer el fichero")

''' This method loads Bayes needed values and adds the to the wrapper class Bayes '''

def loadFileBayes(file,classNameIndex):

	b = Bayes(constants.getN())
	fileHelper = FileHelper()

	try:
		f = fileHelper.openReadOnlyFile(file)
		
		lineas = f.readlines()
		xVector = []

		for linea in lineas:

			xVector = linea.strip("\r\n").split(",")
			className = xVector[classNameIndex-1]
			del xVector[classNameIndex-1]
			xVector = [float(x) for x in xVector]

			b.addX(xVector,className)

		return b
	except:
		print("Error al leer el fichero")

def loadFileTest(file,classNameIndex):
	
	fileHelper = FileHelper()
	try:
		f = fileHelper.openReadOnlyFile(file)
		
		lineas = f.readlines()
		xVector = []

		for linea in lineas:

			xVector = linea.strip("\r\n").split(",")
			del xVector[classNameIndex-1]
			xVector = [float(x) for x in xVector]

		return xVector
	except:
		print("Error al leer el fichero")



if __name__ == "__main__":
	

	test1 = loadFileTest("TestIris01.txt",5)
	print "Loaded test ", test1
	test2 = loadFileTest("TestIris02.txt",5)
	print "Loaded test ", test2
	test3 = loadFileTest("TestIris03.txt",5)
	print "Loaded test ", test3

	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "+++++++++++"
	print "+++++++++++                KMEANS"
	print "+++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"

	k = loadFileKMeans("Iris2Clases.txt",5)

	print ">>> Carga finalizada KMEANS"

	k.doTraining(epsilonLimit=constants.getKMeansEpsilon(),b=constants.getKMeansB())

	print "\n>>> Test KMEANS \n"
	print "Test 1:"
	print  k.clasifyEuclideanDistance(test1),"\n"
	print k.clasifyProbability(test1,constants.getKMeansB())
	print "\nTest 2:"
	print k.clasifyEuclideanDistance(test2),"\n"
	print k.clasifyProbability(test2,constants.getKMeansB())
	print "\nTest 3:"
	print  k.clasifyEuclideanDistance(test3),"\n"
	print k.clasifyProbability(test3,constants.getKMeansB())
	
	
	print "\n\n\n"
	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "+++++++++++"
	print "+++++++++++                BAYES"
	print "+++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"
	print "++++++++++++++++++++++++++++++++++++++++++++"

	bayes = loadFileBayes("Iris2Clases.txt",5)

	print ">>> Carga finalizada BAYES"
	bayes.doTraining()
	classes = bayes.getClasses()
	for value in classes:
		print "\n>>>>>>>>> Clase: ", value
		print "\n>>> M:\n", bayes.getClass(value).getMVector()
		print "\n>>> C:\n", bayes.getClass(value).getCMatrix()
	print "\n>>> Test Bayes \n"
	print "Test 1:"
	print test1," clasificado como clase ", bayes.clasify(test1)
	print "\nTest 2:"
	print test2," clasificado como clase ", bayes.clasify(test2)
	print "\nTest 3:"
	print test3," clasificado como clase ", bayes.clasify(test3)