import numpy as np

class DataManagement:
    def __init__(self,tarps,dataFeatures,dataPhenotypes):
        #About Attributes
        self.numAttributes = dataFeatures.shape[1]
        self.attributeInfoType = [] #False if discrete. True if Continuous
        self.attributeInfo = [] #Array of discrete values if discrete, or Range [Low, High] if continuous
        for i in range(self.numAttributes):
            self.attributeInfoType.append(0)
            self.attributeInfo.append([])

        #About Phenotypes
        self.discretePhenotype = True
        self.phenotypeList = [] # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.isDefault = True  # Is discrete attribute limit an int or string
        try:
            int(tarps.discreteAttributeLimit)
        except:
            self.isDefault = False

        #Split up dataset into list of nparrays by patient instance
        self.trainFormatted = self.formatData(dataFeatures,dataPhenotypes)

        #About Dataset
        self.numTrainInstances = len(self.trainFormatted)
        self.numRows = dataFeatures.shape[0]
        self.discriminatePhenotype(dataPhenotypes, tarps)
        if (self.discretePhenotype):
            self.discriminateClasses(dataPhenotypes)
        else:
            self.characterizePhenotype(dataPhenotypes)

        self.discriminateAttributes(dataFeatures, tarps)
        self.characterizeAttributes(dataFeatures)


    def formatData(self,features,phenotypes):
        formatted = []
        instance = np.array([])
        combined = np.insert(features,self.numAttributes,phenotypes,1)
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
            counter+=1
        np.random.shuffle(formatted)
        return formatted

    def discriminatePhenotype(self, phenotypes, tarps):  # Determine if phenotype is discrete or continuous
        try:
            int(tarps.discretePhenotypeLimit)
            self.isPhenotypeDefault = True
        except:
            self.isPhenotypeDefault = False

        if (self.isPhenotypeDefault):
            currentPhenotypeIndex = 0
            classDict = {}
            while (self.discretePhenotype and len(list(
                    classDict.keys())) <= tarps.discretePhenotypeLimit and currentPhenotypeIndex < self.numRows):
                target = phenotypes[currentPhenotypeIndex]
                if (target in list(classDict.keys())):
                    classDict[target] += 1
                elif np.isnan(target):
                    pass
                else:
                    classDict[target] = 1
                currentPhenotypeIndex += 1

            if (len(list(classDict.keys())) > tarps.discretePhenotypeLimit):
                self.discretePhenotype = False
                self.phenotypeList = [float(target), float(target)]
        elif tarps.discretePhenotypeLimit == "c":
            self.discretePhenotype = False
            self.phenotypeList = [float(phenotypes[0]), float(phenotypes[0])]
        elif tarps.discretePhenotypeLimit == "d":
            self.discretePhenotype = True
            self.phenotypeList = []

    def discriminateClasses(self, phenotypes):
        currentPhenotypeIndex = 0
        classCount = {}
        while (currentPhenotypeIndex < self.numRows):
            target = phenotypes[currentPhenotypeIndex]
            if target in self.phenotypeList:
                classCount[target] += 1
            elif np.isnan(target):
                pass
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
            currentPhenotypeIndex += 1

    def characterizePhenotype(self, phenotypes):
        for target in phenotypes:
            if np.isnan(target):
                pass
            elif float(target) > self.phenotypeList[1]:
                self.phenotypeList[1] = float(target)
            elif float(target) < self.phenotypeList[0]:
                self.phenotypeList[0] = float(target)
            else:
                pass

    def discriminateAttributes(self, features, tarps):
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(self.numAttributes):
            attIsDiscrete = True
            if self.isDefault:
                currentInstanceIndex = 0
                stateDict = {}
                while attIsDiscrete and len(list(
                        stateDict.keys())) <= tarps.discreteAttributeLimit and currentInstanceIndex < self.numRows:
                    target = features[currentInstanceIndex, att]
                    if target in list(stateDict.keys()):
                        stateDict[target] += 1
                    elif np.isnan(target):
                        pass
                    else:
                        stateDict[target] = 1
                    currentInstanceIndex += 1

                if len(list(stateDict.keys())) > tarps.discreteAttributeLimit:
                    attIsDiscrete = False
            elif tarps.discreteAttributeLimit == "c":
                if att in tarps.specifiedAttributes:
                    attIsDiscrete = False
                else:
                    attIsDiscrete = True
            elif tarps.discreteAttributeLimit == "d":
                if att in tarps.specifiedAttributes:
                    attIsDiscrete = True
                else:
                    attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfoType[att] = False
                self.discreteCount += 1
            else:
                self.attributeInfoType[att] = True
                self.continuousCount += 1

    def characterizeAttributes(self, features):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes):
            if self.attributeInfoType[currentFeatureIndexInAttributeInfo]:
                self.attributeInfo[currentFeatureIndexInAttributeInfo] = [float('inf'), float('-inf')]

            for currentInstanceIndex in range(self.numRows):
                target = features[currentInstanceIndex, currentFeatureIndexInAttributeInfo]
                if not self.attributeInfoType[currentFeatureIndexInAttributeInfo]:  # if attribute is discrete
                    if target in self.attributeInfo[currentFeatureIndexInAttributeInfo] or np.isnan(target):
                        pass
                    else:
                        self.attributeInfo[currentFeatureIndexInAttributeInfo].append(target)
                else:  # if attribute is continuous
                    if np.isnan(target):
                        pass
                    elif float(target) > self.attributeInfo[currentFeatureIndexInAttributeInfo][1]:
                        self.attributeInfo[currentFeatureIndexInAttributeInfo][1] = float(target)
                    elif float(target) < self.attributeInfo[currentFeatureIndexInAttributeInfo][0]:
                        self.attributeInfo[currentFeatureIndexInAttributeInfo][0] = float(target)
                    else:
                        pass