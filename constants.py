import numpy 
import random


def getKMeansB():
	return 2

def getLloydsTolerance():
	return numpy.power(10.0,-10)

def getKMeansEpsilon():
	return 0.01

def getLloydsMaxK():
	return 10

def getLLoydsGammaK():
	return 0.1

def getSOMTolerance():
	return numpy.power(10.0,-6)

def getSOMMaxK():
	return 1000

def getSOMGammaK():
	return 0.1

def getSOMInitialAlfa():
	return 0.1

def getSOMFinalAlfa():
	return 0.01

def getSOMDistanceT():
	return numpy.power(10.0,-5)

def getN():
	return 4

def getClassesCount():
	return 2

def getKMeansInitializeUMAtrix(xVectorsCount):

	random.seed()
	ut = []
	print ">>> Numero de vectores de entrenamiento: ",xVectorsCount
	
	for x in range(xVectorsCount):
		rand = random.random()
		iRandom = 1-rand
		ut.append(rand)
		ut.append(iRandom)
	ut = numpy.reshape(ut,(xVectorsCount,getClassesCount()))

	return ut