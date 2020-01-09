from math import log
import operator


# Simple method to create the dataset for our example
def createds():
    ds = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return ds, labels


# Method to calculate shannon entropy
def shannonentropy(dataset):
    dslength = len(dataset)  # Number of rows in dataset
    labels = {}
    for vector in dataset:
        currentlabel = vector[-1]  # Assuming that last column of dataset contains labels
        if currentlabel not in labels.keys():
            labels[currentlabel] = 0
        labels[currentlabel] += 1  # Preparing frequency of each label
    entropy = 0.0
    for key in labels:
        prob = float(labels[key]) / dslength
        entropy -= prob * log(prob, 2)  # Actual entropy calculation
    return entropy


# Method to split the dataset across a feature
# Takes a 2D array, feature position, and feature value as input
def splitds(dataset, axis, value):
    retvalue = []
    for vector in dataset:
        if vector[axis] == value:
            reducedvector = vector[:axis]
            reducedvector.extend(vector[axis + 1:])
            retvalue.append(reducedvector)
    return retvalue


# Method to calculate the best feature to split the dataset on
def bestfeature(ds):
    nfeatures = len(ds[0]) - 1  # Assuming that the last column of dataset contains labels
    baseentropy = shannonentropy(ds)
    bestfeature = -1
    maxgain = 0.0
    for i in range(nfeatures):
        featurevalues = [example[i] for example in ds]
        uniquevals = set(featurevalues)  # Get all unique feature values
        entropy = 0.0
        for value in uniquevals:
            reducedds = splitds(ds, i, value)  # Split across one value
            prob = len(reducedds) / float(len(ds))
            entropy += prob * shannonentropy(reducedds)  # Calculate entropy
        gain = baseentropy - entropy  # Calculate information gain
        if (maxgain < gain):
            maxgain = gain
            bestfeature = i
    return bestfeature  # Return the feature with best information gain


# Method to calculate frequency of feature values and
# return the one with the most frequency
def majority(classlist):
    votes = {}
    for cls in classlist:
        if cls not in votes.keys():
            votes[cls] = 0
        votes[cls] += 1
    sortedvotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedvotes[0][0]


# Method to create a decision tree
# Basic methodology:
# 1. If all classes have same value, return that value
# 2. If there are no more features to split on, return the majority value
# 3. Split the tree on best feature recursively
def createtree(ds, labels):
    classlist = [example[-1] for example in ds]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]  # Implements #1
    if len(ds[0]) == 1:
        return majority(classlist)  # Implements #2
    bestfeat = bestfeature(ds)
    bestfeatlabel = labels[bestfeat]
    tree = {bestfeatlabel: {}}
    del (labels[bestfeat])
    featvalues = [example[bestfeat] for example in ds]
    uniquefeatvalues = set(featvalues)
    for value in uniquefeatvalues:
        sublabels = labels[:]
        tree[bestfeatlabel][value] = createtree(splitds(ds, bestfeat, value), sublabels)
    return tree


# Method to classify a vector using a given decision tree
def classify(tree, labels, vector):
    firststr = tree.keys()[0]  # Get the first feature of the tree
    seconddict = tree[firststr]  # Get the subtree
    featureidx = labels.index(firststr)  # Find out on which index in the data does this feature lie
    for key in seconddict.keys():
        if vector[featureidx] == key:  # If a particular branch of subtree matches corresponding vector's value
            if type(seconddict[key]).__name__=='dict':
                classlabel = classify(seconddict[key], labels, vector)  # Go down the branch if it is not a leaf node
            else:
                classlabel = seconddict[key]  # Return the label if it is a leaf node
    return classlabel


def storetree(tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()


def grabtree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)