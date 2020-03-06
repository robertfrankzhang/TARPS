import pandas as pd
from TARPS import *

data = pd.read_csv("mp.csv",sep=",")
dataFeatures = data.drop("class",axis=1).values
dataPhenotypes = data['class'].values

clf = TARPS()
clf.fit(dataFeatures,dataPhenotypes)
