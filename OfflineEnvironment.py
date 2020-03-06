
from DataManagement import *

class OfflineEnvironment:
    def __init__(self,tarps,features,phenotypes):
        self.dataRef = 0 #Current instance index
        self.storeDataRef = 0
        self.formatData = DataManagement(tarps,features,phenotypes)

        self.currentTrainState = self.formatData.trainFormatted[self.dataRef][:,np.arange(self.formatData.numAttributes)]
        self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][0,self.formatData.numAttributes]

    def getTrainInstance(self):
        return (self.currentTrainState,self.currentTrainPhenotype)

    def newInstance(self):
        if self.dataRef < self.formatData.numTrainInstances - 1:
            self.dataRef += 1
            self.currentTrainState = self.formatData.trainFormatted[self.dataRef][:,np.arange(self.formatData.numAttributes)]
            self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][0, self.formatData.numAttributes]
        else:
            self.resetDataRef()

    def resetDataRef(self):
        self.dataRef = 0
        self.currentTrainState = self.formatData.trainFormatted[self.dataRef][:,np.arange(self.formatData.numAttributes)]
        self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][0, self.formatData.numAttributes]

    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the instance we left off in the training data. """
        self.storeDataRef = self.dataRef

    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        self.dataRef = self.storeDataRef