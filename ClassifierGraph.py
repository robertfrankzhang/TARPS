
import numpy as np
import copy

class ClassifierGraph():
    def __init__(self,maxMacroPopSize):
        self.nextRuleIndex = 0
        self.edgeMatrix = np.zeros(shape=(maxMacroPopSize,maxMacroPopSize))
        self.rules = np.empty(maxMacroPopSize,dtype=object)

    def addUnconnectedRule(self,rule):
        self.rules[self.nextRuleIndex] = rule
        self.edgeMatrix[self.nextRuleIndex,self.nextRuleIndex] = 1 #Make rule self connect
        self.nextRuleIndex += 1

    def duplicateRule(self,rule,ruleIndexToDuplicate): #Duplicate rule and in/outbound edges
        if ruleIndexToDuplicate >= self.nextRuleIndex:
            raise Exception("Cannot duplicate rule that doesn't exist")
        self.rules[self.nextRuleIndex] = copy.deepcopy(self.rules[ruleIndexToDuplicate])

        #Duplicate in/outbound edges
        self.edgeMatrix[:,self.nextRuleIndex] = self.edgeMatrix[:,ruleIndexToDuplicate]
        self.edgeMatrix[self.nextRuleIndex,:] = self.edgeMatrix[ruleIndexToDuplicate,:]

        # Make rule self connect
        self.edgeMatrix[self.nextRuleIndex,self.nextRuleIndex] = 1

        self.nextRuleIndex += 1

    def deleteRule(self,ruleIndexToDelete):
        if ruleIndexToDelete >= self.nextRuleIndex:
            raise Exception("Cannot delete rule that doesn't exist")

        #Shift all following rows up and all following columns left
        for i in range(ruleIndexToDelete,self.nextRuleIndex):
            self.rules[i] = self.rules[i+1]
        for i in range(ruleIndexToDelete,self.nextRuleIndex):
            self.edgeMatrix[:,i] = self.edgeMatrix[:,i+1]
        for i in range(ruleIndexToDelete,self.nextRuleIndex):
            self.edgeMatrix[i,:] = self.edgeMatrix[i+1,:]
        self.nextRuleIndex -= 1

    def addEdge(self,fromRuleAtIndex,toRuleAtIndex):
        if fromRuleAtIndex >= self.nextRuleIndex or toRuleAtIndex >= self.nextRuleIndex:
            raise Exception("Cannot add edge between rule(s) that do not exist")
        self.edgeMatrix[fromRuleAtIndex][toRuleAtIndex] = 1

    def deleteEdge(self,fromRuleAtIndex,toRuleAtIndex):
        if fromRuleAtIndex >= self.nextRuleIndex or toRuleAtIndex >= self.nextRuleIndex:
            raise Exception("Cannot delete edge from rule(s) that do not exist")
        self.edgeMatrix[fromRuleAtIndex][toRuleAtIndex] = 0

