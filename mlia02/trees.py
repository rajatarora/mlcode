from math import log
import operator

def createds():
    ds = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return ds, labels

def shannonentropy(dataset):
    dslength = len(dataset)
    labels = {}
    for vector in dataset:
        currentlabel = vector[-1]
        if currentlabel not in labels.keys():
            labels[currentlabel] = 0
        labels[currentlabel] += 1
    entropy = 0.0
    for key in labels:
        prob = float(labels[key])/dslength
        entropy -= prob * log(prob, 2)
    return entropy

def splitds(dataset, axis, value):
    retvalue = []
    for vector in dataset:
        if vector[axis] == value:
            reducedvector = vector[:axis]
            reducedvector.extend(vector[axis+1:])
            retvalue.append(reducedvector)
    return retvalue

def bestfeature(ds):
    nfeatures = len(ds[0]) - 1
    baseentropy = shannonentropy(ds)
    bestfeature = -1
    maxgain = 0.0
    for i in range(nfeatures):
        featurevalues = [example[i] for example in ds]
        uniquevals = set(featurevalues)
        entropy = 0.0
        for value in uniquevals:
            reducedds = splitds(ds, i, value)
            prob = len(reducedds) / float(len(ds))
            entropy += prob * shannonentropy(reducedds)
        gain = baseentropy - entropy
        if (maxgain < gain):
            maxgain = gain
            bestfeature = i
    return bestfeature

def majority(classlist):
    votes={}
    for cls in classlist:
        if cls not in votes.keys():
            votes[cls] = 0
        votes[cls] += 1
    sortedvotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedvotes[0][0]

