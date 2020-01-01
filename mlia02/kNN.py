from numpy import *
import operator
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
    fr = open(filename) # Open the file for reading
    numberOfLines = len(fr.readlines()) # Find out number of lines
    returnMat = zeros((numberOfLines, 3)) # Create a matrix of zeros: number of lines x number of features
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] # Fill the matrix with first 3 columns in the file
        classLabelVector.append(int(listFromLine[-1])) # Get the class label from the last column of the file
        index +=1
    return returnMat, classLabelVector

def createPlotForEx1():
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

def createPlotForEx2(x, y):
    datingMat, datingLabels = file2matrix("mlia02/datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print datingMat[:,x]
    ax.scatter(datingMat[:,x], datingMat[:,y], 15.0*array(datingLabels), 5.0*array(datingLabels))
    plt.show()


def autonorm(dataset):
    minvals = dataset.min(0) # Get minimum numeric value of each feature. 1D array of no. of features
    maxvals = dataset.max(0) # Get maximum numeric value of each feature. 1D array of no. of features
    ranges = maxvals - minvals # Get (max - min) for each feature. 1D array of no. of features
    normdataset = zeros(shape(dataset)) # Create a '0' matrix of the same shape as dataset
    m = dataset.shape[0] # No. of rows in the dataset
    normdataset = dataset - tile(minvals, (m,1)) # Repeat min values array for the number of rows,
                                                 # then subtract it from dataset
    normdataset = normdataset/tile(ranges, (m,1)) # Divide each row of the dataset with (max - min) range
    return normdataset, ranges, minvals

def datingDataTest(ratio, k):
    # ratio = 0.15
    datingMat, datingLabels = file2matrix("mlia02/datingTestSet2.txt")
    normMat, ranges, minvals = autonorm(datingMat)
    m = normMat.shape[0]
    numTestVecs = int(m*ratio)
    errorcount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], k)
        print "Classified: %d, Actual: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorcount += 1.0
    print "Total error rate: %f" % (errorcount/float(numTestVecs))