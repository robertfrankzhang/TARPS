
import pandas as pd
import numpy as np
import random

'''
This can perform CV and data setup/filtering for TARPS and standard model fit/score/predict. Regular CV doesn't work
for TARPS because of how data is formatted.

It also also test data to be stratified by missingness for both TARPS and nonTARPS data, which also A/B performance
testing between TARPS and other models to be possible.
'''

class DataSetup:
    def __init__(self,filename,classLabel,isTARPS):
        #Assumes data is fully numeric or NaN
        self.classLabel = classLabel
        data = pd.read_csv(filename,sep=',')
        self.dataFeatures = data.drop(classLabel, axis=1).values  # splits into an array of instances
        self.dataPhenotypes = data[classLabel].values
        self.dataHeaders = data.drop(classLabel, axis=1).columns.values

        self.numTARPSInstances = None #Number of TARPS patient instances, None if not TARPS
        self.numInstances = 0
        self.deleteAllInstancesWithoutPhenotype(isTARPS)
        self.isTARPS = isTARPS

    def deleteAllInstancesWithoutPhenotype(self,isTARPS):
        newFeatures = np.array([[2, 3]])
        newPhenotypes = np.array([])
        firstTime = True

        if isTARPS:
            self.numTARPSInstances = 0

        for instanceIndex in range(len(self.dataFeatures)):
            instance = self.dataPhenotypes[instanceIndex]
            if not np.isnan(instance):
                self.numInstances+=1
                if firstTime:
                    firstTime = False
                    newFeatures = np.array([self.dataFeatures[instanceIndex]])
                else:
                    newFeatures = np.concatenate((newFeatures, [self.dataFeatures[instanceIndex]]), axis=0)
                newPhenotypes = np.append(newPhenotypes, instance)
            else:
                isValid = False
                for attr in self.dataFeatures[instanceIndex]:
                    if not np.isnan(attr):
                        isValid = True

                if not isValid and isTARPS:#If the line is fully blank and TARPS is true, preserve
                    self.numInstances += 1
                    self.numTARPSInstances += 1
                    if firstTime:
                        firstTime = False
                        newFeatures = np.array([self.dataFeatures[instanceIndex]])
                    else:
                        newFeatures = np.concatenate((newFeatures, [self.dataFeatures[instanceIndex]]), axis=0)
                        newPhenotypes = np.append(newPhenotypes, instance)

        self.dataFeatures = newFeatures
        self.dataPhenotypes = newPhenotypes

    def getTrainingAndTestingData(self,cv=1): #If cv = 1, returns all data as training
        if self.isTARPS:
            numInstances = self.numTARPSInstances
        else:
            numInstances = self.numInstances

        if cv == 1:
            trainingDataSize = numInstances
        else:
            trainingDataSize = numInstances - int(numInstances/cv)
        trainingIndexes = random.sample(range(numInstances),trainingDataSize)
        testingIndexes = []
        for i in range(numInstances):
            if not i in trainingIndexes:
                testingIndexes.append(i)
        testingIndexes = random.sample(testingIndexes,len(testingIndexes))
        print(trainingIndexes)
        print(testingIndexes)

        if not self.isTARPS:
            self.trainingData = []
            self.trainingPhenotypes = []
            self.testingData = []
            self.testingPhenotypes = []
            for i in range(numInstances):
                if i in trainingIndexes:
                    self.trainingData.append(self.dataFeatures[i])
                    self.trainingPhenotypes.append(self.dataPhenotypes[i])
                if i in testingIndexes:
                    self.testingData.append(self.dataFeatures[i])
                    self.testingPhenotypes.append(self.dataPhenotypes[i])

        else:
            formatted = []
            instance = np.array([])
            combined = np.insert(self.dataFeatures, self.dataFeatures.shape[1], self.dataPhenotypes, 1)
            counter = 0
            for snapshot in combined:
                isValid = False
                for attr in snapshot:
                    if not np.isnan(attr):
                        isValid = True
                if isValid:
                    if counter == 0:
                        instance = np.array([snapshot])
                    else:
                        instance = np.concatenate((instance, [snapshot]), axis=0)
                else:
                    formatted.append(instance)
                    instance = np.array([])
                    counter = -1
                counter += 1

            self.trainingData = []
            self.trainingPhenotypes = []
            self.testingData = []
            self.testingPhenotypes = []

            for i in range(numInstances):
                if i in trainingIndexes:
                    trainInst = formatted[i][:,np.arange(self.dataFeatures.shape[1])] #ndarray of snapshots
                    trainPheno = formatted[i][0, self.dataFeatures.shape[1]]
                    numSnaps = len(formatted[i][:,np.arange(self.dataFeatures.shape[1])])
                    emptyRow = np.array([np.nan]*self.dataFeatures.shape[1])
                    for inst in trainInst:
                        self.trainingData.append(inst)
                        self.trainingPhenotypes.append(trainPheno)
                    self.trainingData.append(emptyRow)
                    self.trainingPhenotypes.append(np.nan)

                if i in testingIndexes:
                    testInst = formatted[i][:, np.arange(self.dataFeatures.shape[1])]  # ndarray of snapshots
                    testPheno = formatted[i][0, self.dataFeatures.shape[1]]
                    numSnaps = len(formatted[i][:, np.arange(self.dataFeatures.shape[1])])
                    emptyRow = np.array([np.nan]*self.dataFeatures.shape[1])
                    for finalSnapIndex in range(1, numSnaps):
                        for index in range(0, finalSnapIndex):
                            self.testingData.append(testInst[index])
                            self.testingPhenotypes.append(testPheno)
                        self.testingData.append(emptyRow)
                        self.testingPhenotypes.append(np.nan)

        self.trainingData = np.array(self.trainingData)
        self.trainingPhenotypes = np.array(self.trainingPhenotypes)
        self.testingData = np.array(self.testingData)
        self.testingPhenotypes = np.array(self.testingPhenotypes)
        return self.trainingData, self.trainingPhenotypes, self.testingData, self.testingPhenotypes


    def separateTestingStrata(self,testingFeatures,testingPhenotypes):
        #Params formatted as 2D/1D ndarray w/ spaces in between instances if TARPS, no spaces if not TARPS
        #Separates testing into missing strata instances.
        testingStrata = {} #Dictionary of keys, where keys are the # of missing attributes in feature.
        justFinishedBlankLine = True
        if self.isTARPS:
            #Separates each patient instance into a bunch of paths of snapshots via working backwards. Missingness determined by most recent snapshot missingness
            for instIndex in range(len(testingFeatures)):
                emptyRow = np.array([np.nan] * testingFeatures.shape[1])
                if justFinishedBlankLine:
                    beginIndex = instIndex
                    justFinishedBlankLine = False
                isBlank = True
                for attr in testingFeatures[instIndex]:
                    if not np.isnan(attr):
                        isBlank = False

                if isBlank:
                    justFinishedBlankLine = True
                    endIndex = instIndex
                    numMissing = 0
                    for attr in testingFeatures[instIndex-1]:
                        if np.isnan(attr):
                            numMissing += 1
                    if numMissing in testingStrata.keys():
                        for r in range(beginIndex, endIndex):
                            testingStrata[numMissing][0].append(testingFeatures[r])
                            testingStrata[numMissing][1].append(testingPhenotypes[r])
                    else:
                        testingStrata[numMissing] = ([testingFeatures[beginIndex]],[testingPhenotypes[beginIndex]])
                        for r in range(beginIndex+1,endIndex):
                            testingStrata[numMissing][0].append(testingFeatures[r])
                            testingStrata[numMissing][1].append(testingPhenotypes[r])
                    testingStrata[numMissing][0].append(emptyRow)
                    testingStrata[numMissing][1].append(np.nan)

        else:
            #Separates each instance by missingnness
            for instIndex in range(len(testingFeatures)):
                numMissing = 0
                for attr in testingFeatures[instIndex]:
                    if np.isnan(attr):
                        numMissing+=1
                if numMissing in testingStrata.keys():
                    testingStrata[numMissing][0].append(testingFeatures[instIndex])
                    testingStrata[numMissing][1].append(testingPhenotypes[instIndex])
                else:
                    testingStrata[numMissing] = ([testingFeatures[instIndex]],[testingPhenotypes[instIndex]])

        for k in testingStrata:
            testingStrata[k] = (np.array(testingStrata[k][0]),np.array(testingStrata[k][1]))

        return testingStrata #Should return a usable test set (features,phenotypes) for each level of missingness


