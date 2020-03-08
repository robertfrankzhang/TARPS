import pandas as pd
from TARPS import *
from DataSetup import *

storage = DataSetup("mp.csv","class",True)
trainingData, trainingPhenotypes, testingData, testingPhenotypes = storage.getTrainingAndTestingData(cv=3)
strata = storage.separateTestingStrata(testingData,testingPhenotypes)

# data = pd.read_csv("mp.csv",sep=",")
# dataFeatures = data.drop("class",axis=1).values
# dataPhenotypes = data['class'].values
#
# clf = TARPS()
# clf.fit(dataFeatures,dataPhenotypes)
