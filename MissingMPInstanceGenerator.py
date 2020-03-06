
import random
import copy
import csv
'''
This class creates exports a csv of shuffled missing n-bit multiplexer instances. A missing multiplexer instance is a
multiclass multiplexer (i.e. register bits are multiclass) whose attributes appear in order. The csv writes a blank row
for spaces between instances
'''

class MissingMPInstanceGenerator:
    def __init__(self,bits=6,classCount=4):
        #Check if bits is valid from 6 bit to 135 bit MP
        isValid = False
        addressBitCount = 0
        registerBitCount = 0
        for i in range(2,8):
            if bits == i+pow(2,i):
                isValid = True
                addressBitCount = i
                registerBitCount = bits - addressBitCount

        if not isValid:
            raise Exception("# of bits is invalid. Must be n+2^n, 2 <= n <= 7")

        #Check if classCount is valid
        if classCount < 2:
            raise Exception("# of classes must be at least 2")

        #Populate list of possible MP conditions and actions
        mps = []
        for i in range(pow(2,addressBitCount)):
            address = str(self.baseTenToBase(i,2))
            for j in range(pow(classCount,registerBitCount)):
                register = str(self.baseTenToBase(j,classCount))
                while len(register) < registerBitCount:
                    register = "0"+register
                while len(address) < addressBitCount:
                    address = "0"+address
                nAddress = []
                nRegister = []
                for z in address:
                    nAddress.append(int(z))
                for z in register:
                    nRegister.append(int(z))
                r = self.binaryToDecimal(int(address))
                phenotype = register[r]
                mps.append(nAddress+nRegister+[int(phenotype)])

        #Sort mps by class
        splitMPs = []
        for i in range(classCount):
            splitMPs.append([])

        for i in mps:
            splitMPs[i[len(i)-1]].append(i)

        #Determine appearance order for each class
        shuffleIndices = []
        indices = []
        for i in range(bits):
            indices.append(i)
        for i in range(classCount):
            t = random.sample(indices,len(indices))
            while t in shuffleIndices:
                t = random.sample(indices, len(indices))
            shuffleIndices.append(t)

        #Create instances
        self.instances = []
        for classIndex in range(classCount):
            for inst in splitMPs[classIndex]:
                newInst = [""]*(bits+1)
                newInst[bits] = inst[bits] #Phenotype
                singleInst = []
                for appearIndex in shuffleIndices[classIndex]:
                    newInst[appearIndex] = inst[appearIndex]
                    singleInst.append(copy.deepcopy(newInst))
                self.instances.append(singleInst)
        self.bits = bits
        self.classCount = classCount

    #Returns num in base endBase
    def baseTenToBase(self,num, endBase):
        s = ""
        while int(num / endBase) != 0:
            r = num % endBase
            num = int(num / endBase)
            s = str(r) + s
        r = num % endBase
        s = str(r) + s
        return int(s)

    def binaryToDecimal(self,binary):
        binary1 = binary
        decimal, i, n = 0, 0, 0
        while (binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary // 10
            i += 1
        return decimal

    def exportInstances(self,filename="mp.csv"):
        headerNames = []
        for i in range(self.bits+1):
            if i != self.bits:
                headerNames.append("N"+str(i))
            else:
                headerNames.append("class")
        with open(filename,mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headerNames)
            for inst in self.instances:
                for snapshot in inst:
                    writer.writerow(snapshot)
                writer.writerow([""]*(self.bits+1))


m = MissingMPInstanceGenerator(bits=6,classCount=4)
m.exportInstances()