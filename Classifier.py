
import random

class Classifier:
    def __init__(self,tarps):
        self.specifiedAttList = []
        self.condition = []
        self.phenotype = None

        self.fitness = tarps.init_fit
        self.accuracy = 0.0
        self.aveMatchSetSize = None
        self.deletionVote = None
        self.deletionProb = None

        self.timeStampGA = None
        self.initTimeStamp = None

        self.matchCount = 0
        self.correctCount = 0