import basicLogisticRegression
import rmp
import numpy as np
import pandas as pd

#    xtrain, ytrain, xvalid, yvalid, xtest = readData()
#    xtrain, ytrain, xvalid, yvalid, xtest = cleanData(xtrain, ytrain, xvalid, yvalid, xtest)
xtrain, ytrain, xvalid, yvalid, xtest = rmp.readData()
#    xtrain = xtrain[0:30000]
#    ytrain = ytrain[0:30000]
xtrain, xvalid, xtest, countVect = rmp.vectorizeWords(xtrain, xvalid, xtest)

ytest, w, b = basicLogisticRegression.resumeLogisticRegression(xtrain, ytrain, xvalid, yvalid, xtest, 1)
#1000 epochs = 30 min

ytest = pd.Series(ytest, index=xtest.iloc[:,1])


