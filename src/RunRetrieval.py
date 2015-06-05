#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import pdb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import correlate2d
import pprint

# import matplotlib.pyplot as plt

from matplotlib.pyplot import *
import Retrieval as ret
from Parameters import *
import matplotlib as mpl
import copy
import sys

mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['font.size'] = 25
FeaturePath = sys.argv[1]

###### Define Paths #######

dataBaseFile0 = FeaturePath + 'Features_level_0.txt'
userInputFile0 = FeaturePath + 'UserFeatures_level_0.txt'

dataBaseFile1 = FeaturePath + 'Features_level_1.txt'
userInputFile1 = FeaturePath + 'UserFeatures_level_1.txt'

dataBaseFile2 = FeaturePath + 'Features_level_2.txt'
userInputFile2 = FeaturePath + 'UserFeatures_level_2.txt'

dataBaseFile3 = FeaturePath + 'Features_level_3.txt'
userInputFile3 = FeaturePath + 'UserFeatures_level_3.txt'

dataBaseFile4 = FeaturePath + 'Features_level_4.txt'
userInputFile4 = FeaturePath + 'UserFeatures_level_4.txt'

dataBaseFile5 = FeaturePath + 'Features_level_5.txt'
userInputFile5 = FeaturePath + 'UserFeatures_level_5.txt'

####### Define variables #####

dataBase0 = []
dataBase1 = []
dataBase2 = []
dataBase3 = []
dataBase4 = []
dataBase5 = []

labels = []

queryList0 = []
queryList1 = []
queryList2 = []
queryList3 = []
queryList4 = []
queryList5 = []
qLabels = []
uId = []
p = Parameters()

###### Read User Inputs and dataBase ########

with open(dataBaseFile0) as f:
    for line in f:
        elements = line.strip('\n').split('#')

        label = elements[1].replace('#', '_')
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        if len(feature) == 0:
            pdb.set_trace()
            continue
        labels.append(label)
        dataBase0.append(feature)

with open(userInputFile0) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        uId.append(elements[0])
        label = elements[1].split('.')[0].replace('#', '_')
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        qLabels.append(label)
        queryList0.append(feature)

with open(dataBaseFile1) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        label = elements[1].replace('#', '_')
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        dataBase1.append(feature)

with open(userInputFile1) as f:
    for line in f:
        elements = line.strip('\n').split('#')

        label = elements[1].split('.')[0].replace('#', '_')
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        queryList1.append(feature)

with open(dataBaseFile2) as f:
    for line in f:
        elements = line.strip('\n').split('#')

        label = elements[1].replace('#', '_')
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        if len(feature) == 0:
            dataBase2.append('Empty')
        else:
            dataBase2.append(feature)

with open(userInputFile2) as f:
    for line in f:
        elements = line.strip('\n').split('#')

        label = elements[1].split('.')[0].replace('#', '_')
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        queryList2.append(feature)

with open(dataBaseFile3) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        label = elements[1].split('.')[0]
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        dataBase3.append(feature)

with open(userInputFile3) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        label = elements[1].split('.')[0]
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        queryList3.append(feature)

with open(dataBaseFile4) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        label = elements[1].split('.')[0]
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        dataBase4.append(np.array(feature))

with open(userInputFile4) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        label = elements[1].split('.')[0]
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        queryList4.append(np.array(feature))

with open(dataBaseFile5) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        label = elements[1].split('.')[0]
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        dataBase5.append(np.array(feature))

with open(userInputFile5) as f:
    for line in f:
        elements = line.strip('\n').split('#')
        label = elements[1].split('.')[0]
        feature = elements[3]
        feature = [i for i in feature.split(',')]
        feature.pop()
        feature = [float(i) for i in feature]
        queryList5.append(np.array(feature))

                # ###################### Cascaded Search ##################

precision0 = []
precision1 = []
precision2 = []
precision2_5 = []
precision3 = []
precision4 = []
precision4_5 = []
precision5 = []
recall0 = []
recall1 = []
recall2 = []
recall2_5 = []
recall3 = []
recall4 = []
recall4_5 = []
recall5 = []
accuracy0 = 0
accuracy1 = 0
accuracy2 = 0
accuracy3 = 0
accuracy4 = 0
accuracy5 = np.zeros(p.topk)
ommitted = 0
mRR = []

count = 0
for (
    us,
    qLabel,
    query0,
    query1,
    query2,
    query3,
    query4,
    query5,
    ) in zip(
    uId,
    qLabels,
    queryList0,
    queryList1,
    queryList2,
    queryList3,
    queryList4,
    queryList5,
    ):

    if count == 100000:
        break
    count += 1
    print(us, qLabel, end=' ')
    retLab0 = []
    retLab1 = []
    retLab2 = []
    retLab3 = []
    retLab4 = []
    retLab5 = []
    score = np.zeros(len(labels))
    newDB2 = []
    newDb3 = []
    newDb4 = []
    newDb5 = []

    # ######## Ground Truth ############
    # Compares the raw X,Y points of extracted trajectories and sketches

    (indexes_0, dis_0) = ret.doDtw(query0, dataBase0, labels, p,
                                   range(0, len(labels)))
    for i in indexes_0:
        retLab0.append(labels[i])

    (precisionArr0, recallArr0) = ret.findPrecisionRecall(qLabel,
            retLab0)
    precision0.append(precisionArr0)
    recall0.append(recallArr0)

    # ######## Level 1 #########
    # Compares the histogram representation of the trajectories and sketches

    (indexes_1, dis_1) = ret.levelOneFiltering(query1, dataBase1,
            labels, p)
    dis_1 = dis_1 / np.amax(dis_1)
    for (d, idx) in zip(dis_1[0:p.l1c], (indexes_1[0])[0:p.l1c]):
        retLab1.append(labels[idx])
        newDB2.append(dataBase2[idx])
    score = ret.updateScore(score, labels, (indexes_1[0])[0:p.l1c],
                            dis_1)

    (precisionArr1, recallArr1) = ret.findPrecisionRecall(qLabel,
            retLab1)
    precision1.append(precisionArr1)
    recall1.append(recallArr1)

    if qLabel in retLab1[0:20]:
        accuracy1 += 1.0

    # ##########   Level 2  #############
    # Compares the circle based representations of the ordered motion and sketch segments using a DTW distance metric.

    newnewDB2 = []
    newretLab1 = []
    newIndexes = []
    if len(query2) == 0:
        continue
    for (index, data, lab) in zip((indexes_1[0])[0:p.l1c], newDB2,
                                  retLab1):
        if data == 'Empty':
            score[index] = 1000
            continue
        else:
            newnewDB2.append(data)
            newretLab1.append(lab)
            newIndexes.append(index)
    newDB2 = newnewDB2
    retLab1 = newretLab1

    (indexes_2, dis_2) = ret.doDtw(query2, newDB2, retLab1, p,
                                   newIndexes)

    dis_2 = dis_2 / np.amax(dis_2)

    for (d, idx) in zip(dis_2[0:p.l2c], indexes_2[0:p.l2c]):
        retLab2.append(labels[idx])

        newDb4.append(dataBase4[idx])

    score = ret.updateScore(score, labels, indexes_2[0:p.l2c], dis_2)

    (precisionArr2, recallArr2) = ret.findPrecisionRecall(qLabel,
            retLab2)

    precision2.append(precisionArr2)
    recall2.append(recallArr2)

    newLabels = copy.deepcopy(labels)
    newScore = copy.deepcopy(score)

    (newScore, newLabels) = (list(x) for x in zip(*sorted(zip(newScore,
            newLabels))))

    (precisionArr2_5, recallArr2_5) = ret.findPrecisionRecall(qLabel,
            newLabels[0:p.l2c])

    precision2_5.append(precisionArr2_5)
    recall2_5.append(recallArr2_5)

    if qLabel in retLab2[0:20]:

        accuracy2 += 1.0

    # ###########################    Level 4     #####################
    # Compares the change in direction of the trajectory and the online sketches using a DTW distance metric.

    (indexes_4, dis_4) = ret.doDtw(query4, newDb4, retLab2, p,
                                   indexes_2)
    dis_4 = dis_4 / np.amax(dis_4)

    for (d, idx) in zip(dis_4[0:p.l4c], indexes_4[0:p.l4c]):
        retLab4.append(labels[idx])

        newDb5.append(dataBase5[idx])

    score = ret.updateScore(score, labels, indexes_4[0:p.l4c], dis_4)

    (precisionArr4, recallArr4) = ret.findPrecisionRecall(qLabel,
            retLab4)

    precision4.append(precisionArr4)
    recall4.append(recallArr4)

    newLabels = copy.deepcopy(labels)
    newScore = copy.deepcopy(score)

    (newScore, newLabels) = (list(x) for x in zip(*sorted(zip(newScore,
            newLabels))))

    (precisionArr4_5, recallArr4_5) = ret.findPrecisionRecall(qLabel,
            newLabels[0:p.l4_5c])

    precision4_5.append(precisionArr4_5)
    recall4_5.append(recallArr4_5)

    # ######################   Fifth Level ################
    # Compares the scale change of the motion trajectories and the sketch

    (indexes_5, dis_5) = ret.doDtw(query5, newDb5, retLab4, p,
                                   indexes_4)

    dis_5 = dis_5 / np.amax(dis_5)

    for (d, idx) in zip(dis_5[0:p.l5c], indexes_5[0:p.l5c]):
        retLab5.append(labels[idx])
    score = ret.updateScore(score, labels, indexes_5[0:p.l5c], dis_5)

    newLabels = copy.deepcopy(labels)
    newScore = copy.deepcopy(score)

    (newScore, newLabels) = (list(x) for x in zip(*sorted(zip(newScore,
            newLabels))))
    (precisionArr5, recallArr5) = ret.findPrecisionRecall(qLabel,
            newLabels[0:p.l5c])

    precision5.append(precisionArr5)
    recall5.append(recallArr5)
    mRR.append(1.0 / (newLabels.index(qLabel) + 1))
    print(newLabels.index(qLabel) + 1)
    if qLabel in newLabels[0:p.topk]:
        ind = newLabels.index(qLabel)
        for ind1 in range(ind, p.topk):
            accuracy5[ind1] += 1

# print (count);

meanPrecision0 = np.mean(np.array(precision0), 0)
meanRecall0 = np.mean(np.array(recall0), 0)
meanPrecision1 = np.mean(np.array(precision1), 0)
meanRecall1 = np.mean(np.array(recall1), 0)
meanPrecision2 = np.mean(np.array(precision2), 0)
meanRecall2 = np.mean(np.array(recall2), 0)

meanPrecision2_5 = np.mean(np.array(precision2_5), 0)
meanRecall2_5 = np.mean(np.array(recall2_5), 0)
meanPrecision4 = np.mean(np.array(precision4), 0)
meanRecall4 = np.mean(np.array(recall4), 0)
meanPrecision4_5 = np.mean(np.array(precision4_5), 0)
meanRecall4_5 = np.mean(np.array(recall4_5), 0)
meanPrecision5 = np.mean(np.array(precision5), 0)
meanRecall5 = np.mean(np.array(recall5), 0)

accuracy1 = accuracy1 / len(qLabels) * 100
accuracy2 = accuracy2 / len(qLabels) * 100

accuracy5 = np.ceil(accuracy5 / count * 100)
accuracy5 = [int(i) for i in list(accuracy5)]
mRR1 = np.mean(np.array(mRR))

# print (accuracy1,accuracy2,accuracy5,mRR1,count);

print('Mean Reciprocal Rank : ', mRR1)
(hist, bins) = np.histogram(mRR, 75, (0, 1))

plot(meanRecall0, meanPrecision0, 'c', label='Simple DTW')
plot(meanRecall1, meanPrecision1, 'r', label='First Level')

plot(meanRecall2_5, meanPrecision2_5, 'b', label='Second Level')

plot(meanRecall4_5, meanPrecision4_5, 'g', label='Third Level')
plot(meanRecall5, meanPrecision5, 'k', label='Fourth Level')
legend(bbox_to_anchor=(.6, .6, .4, .4), loc=1, ncol=1, mode='expand',
       borderaxespad=0.)

xlabel('Recall', fontsize=25)
ylabel('Precision', fontsize=25)
ylim([0, 1])
xlim([0, 1])
figure()
bar(bins[0:-1], hist, width=0.015)
xlim([0, 1.05])
ylim([0, 75])
xlabel('Reciprocal Ranks', fontsize=25)
ylabel('Number of Queries', fontsize=25)
figure()
plot(range(1, 16), accuracy5, 'r')
plot(range(1, 16), accuracy5, 'bo')
for (x, y, z) in zip(range(1, 16), accuracy5, accuracy5):
    annotate('{}'.format(z), xy=(x, y), xytext=(0, 10), ha='right',
             textcoords='offset points')
ylim([0, 100])
xlim([0, 16])
xlabel('Top K retrievals', fontsize=25)
ylabel('Accuracy', fontsize=25)

show()


			
