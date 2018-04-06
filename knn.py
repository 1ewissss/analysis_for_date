from numpy import *
import operator

def creatdataset():
    group = array([[1,1.1],[1,1],[0,0],[0,1]])
    labels = ['a','a','b','b']
    return group, labels

def classify0(inX, dataset, labels,k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inX, (datasetsize, 1)) - dataset
    sq_diffmat = diffmat**2
    sq_distances = sq_diffmat.sum(axis=1)
    distances = sq_distances ** 0.5
    sortedistindicis = distances.argsort()
    classcount={}
    for i in range(k):
        votelabel = labels[sortedistindicis[i]]
        classcount[votelabel] = classcount.get(votelabel, 0) + 1
    sortedclasscount = sorted(classcount.items(),
                                  key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]
data1,data2=creatdataset()
final=classify0([0.1, 3], data1, data2, 3)
print(final)