import rmp
import theano
import theano.tensor as T
import numpy as np

class LogisticRegression:
    def __init(self, x, nIn, nOut):
        self.W = theano.shared(
            value = np.zeros((nIn, nOut), dtype = theano.config.floatX),
            name = 'W',
            borrow = True
        )
        self.b = theano.shared(
            value = np.zeros((nOut,), dtype = theano.config.floatX),
            name = 'b',
            borrow = True
        )
        self.P_y = T.nnet.softmax(T.dot(x, self.W) + self.b)
        
        self.yHat = T.argmax(self.P_y, axis = 1)
        
        self.params = [self.W, self.b]
        
        self.x = x

    def categorical_crossentropy(self, y):
        return T.nnet.categorical_crossentropy(self.P_y, y)
        
    def error(self, y):
        if y.ndim != self.yHat.ndim:
            raise TypeError('y should have the same shape as yHat', ('y', y.type, 'yHat', self.yHat.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(y, self.yHat)
        else 
            raise NotImplementedError()
            
def loadRMPData(useCleaned):
    if useCleaned:       
        xtrain, ytrain, xvalid, yvalid, xtest = rmp.readCleanData()
    else:
        xtrain, ytrain, xvalid, yvalid, xtest = readData()
        xtrain, ytrain, xvalid, yvalid, xtest = cleanData(xtrain, ytrain, xvalid, yvalid, xtest)        
    xtrain, xvalid, xtest, countVect = rmp.vectorizeWords(xtrain, xvalid, xtest)
    ytest = np.zeros((len(xtest),) dtype = theano.config.floatX)
    
    def shareData(dataX, dataY, borrow=True):
        sharedX = theano.shared(np.asarray(dataX, dtype = theano.config.floatX), borrow=True)
        sharedY = theano.shared(np.asarray(dataY, dtype = theano.config.floatX), borrow=True)
        return sharedX, T.cast(sharedY, 'int32')
    
    xtrain, ytrain = shareData(xtrain, ytrain)
    xvalid, yvalid = shareData(xvalid, yvalid)
    xtest, ytest = shareData(xtest, ytest)
    return [(xtrain, ytrain), (xvalid, yvalid), (xtest, ytest)]
    
def RMPSGD(learningRate=.13, epochs=1000, useCleanedData=True, batchSize=600, regularization=.001):
    datasets = loadRMPData(usedCleanedData)
    
    xtrain, ytrain = datasets[0]
    xvalid, yvalid = datasets[1]
    xtest, ytest = datasets[2]
    
    trainBatches = xtrain.get_value(borrow=True).shape[0] / batchSize
    validBatches = xvalid.get_value(borrow=True).shape[0] / batchSize
    testBatches = xtest.get_value(borrow=True).shape[0] / batchSize
    
    print '... building the model'
    
    index = T.lscalar()
    
    x = T.matrix('x')
    y = T.ivector('y')
    
    classifier = LogisticRegression(x, xtrain.get_value(borrow=True).shape[1], 9)
    
    cost = classifier.categorical_crossentropy - regularization * (w**2).mean()
    
#    testModel = theano.function(
#        inputs=[index],
#        outputs=classifier.error(y),
#        givens={
#            x: xtest[index * batchSize : (index+1)*batchSize],
#            y: ytest[index * batchSize : (index+1)*batchSize]
#        }
#    )

    validateModel = theano.function(
        inputs=[index],
        outputs=classifier.error(y),
        givens={
            x: xvalid[index * batchSize : (index+1) * batchSize],
            y: yvalid[index * batchSize : (index+1) * batchSize]
        }
    )
    
    gw, gb = T.grad(cost, [classifier.W, classifier.b])
    
    trainUpdates = [(classifier.W, classifier.W - learningRate * gw), (classifier.b, classifier.b - learningRate * gb)]
    
    trainModel = theano.function(
        inputs = [index],
        outpus = cost,
        updates = trainUpdates,
        givens = {
            x: xtrain[index * batchSize : (index+1) * batchSize],
            y: ytrain[index * batchSize : (index+1) * batchSize]
        }
    )
    
    
    
    
    
    
