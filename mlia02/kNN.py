from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

# Create a training set of 2D points
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# KNN algorithm
# inX = 2D point to classify
# dataset = 2D points in the training set
# labels = labels for 'dataset'
# k = the k in 'k-nearest neighbours'
def classify0(inX, dataset, labels, k):

    # Distance calculation of inX from every point in the set
    # The matrix way
    datasetSize = dataset.shape[0]
    diffMat = tile(inX, (datasetSize,1)) - dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistances = distances.argsort() # Sorting the distances

    # Read the top 'k' items in the sorted distances
    # Prepare a frequency array for each label in the set
    # Sort desc by frequency
    # Return the topmost frequency
    classCount = {}
    for i in range(k):
        vote = labels[sortedDistances[i]]
        classCount[vote] = classCount.get(vote,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index +=1
    return returnMat, classLabelVector

def createPlot():
    group, labels = createDataSet()
    plt.clf()
    plt.scatter(group[:,0],group[:,1])

    for x in range(0, group.shape[0]):
        print group[x][0], group[x][1]
        plt.annotate(labels[x],
                     (group[x][0], group[x][1]),
                     textcoords="offset points",
                     xytext=(8,-3))

    plt.show()