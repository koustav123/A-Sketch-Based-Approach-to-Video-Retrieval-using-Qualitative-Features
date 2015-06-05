from __future__ import print_function
import pdb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import correlate2d
import pprint
import matplotlib.pyplot as plt
import mlpy
from collections import Counter

def newFindMatch(a,b):
	matchValues=[]
	if len(a)<len(b):
		a,b=b,a;
	
	padding=np.zeros(len(b));
	a=np.hstack((padding,a,padding));
	
	for i in range(len(a)):
		if i==len(a)-len(b)+1:
			break 
		x=a[i:i+len(b)]
		matchValues.append(np.std(x-b))
	return np.array(matchValues);
	
def doDtw(query,dataBase,dataLabels,p,dataLabelIndex):
	dis=[]
	mydtw = mlpy.Dtw(onlydist=True)

	for data in dataBase:
		dist = mydtw.compute(query, data)
		dis.append(dist)

	dis,dataLabelIndex = (list(x) for x in zip(*sorted(zip(dis,dataLabelIndex))));
	return(dataLabelIndex,np.array(dis));
	
def levelOneFiltering(query,dataBase,dataLabels,p):
	neigh = KNeighborsClassifier(n_neighbors=p.nn,metric='euclidean',)
	neigh.fit(np.array(dataBase),dataLabels);
	pDis,pLabels=neigh.kneighbors(query)
	return pLabels,pDis[0];

	
def levelTwoFiltering(query,qLabel,dataBase,dataLabels,p):
	dis=[]
	figureCount=1
	for c,(data,oLabel) in enumerate(zip(dataBase,dataLabels)):
		if (c)%5==0:
			plt.figure(figureCount);figureCount+=1;
			plt.subplot(6,2,1);plt.title(qLabel);plt.bar(xrange(len(query)),query);plt.xlim([0,200]);plt.ylim([-2,2]);			
		xlen=len(data)+len(query);
		plt.subplot(6,2,2*(c%5+1)+1);plt.title(oLabel);plt.bar(xrange(len(data)),data);plt.xlim([0,200]);plt.ylim([-2,2]);
		conv=newFindMatch(np.array(query),np.array(data));
		dis.append(np.amin(conv));
		plt.subplot(6,2,2*(c%5+1)+2);plt.title("Variance Match");plt.bar(xrange(len(conv)),conv);plt.xlim([0,200]);plt.ylim([-2,2]);
		
	plt.tight_layout();
	plt.close();
	return(np.array(dis))
	
def levelThreeFiltering(query,dataBase,dataLabels,p,dataLabelIndex):
	dis=[];
	for c,(data,label) in enumerate(zip(dataBase,dataLabels)):
		if len(data)<len(query):
			data,query=query,data;
	
		padding=np.zeros(len(data)-len(query));
		query=np.hstack((padding,query));
		
		z=mlab.xcorr(query,data,'coeff');
		dis.append(max(z));
	dis,dataLabelIndex = (list(x) for x in zip(*sorted(zip(dis,dataLabelIndex))));
	return(dataLabelIndex,np.array(dis));	
	
def levelTwoFiltering2d(query,qLabel,dataBase2,dataBase3,dataLabels,p):
	dis=[];
	figureCount=1
	for c,(data2,data3,oLabel) in enumerate(zip(dataBase2,dataBase3,dataLabels)):
		
		if (c)%3==0:
			plt.figure(figureCount);figureCount+=1;
			plt.subplot(4,3,1);plt.title(qLabel);plt.bar(xrange(len(query[0])),query[0]);plt.xlim([0,100]);plt.ylim([-2,2]);
			plt.subplot(4,3,2);plt.title(qLabel);plt.bar(xrange(len(query[1])),query[1]);plt.xlim([0,100]);plt.ylim([-2,2]);
			
		plt.subplot(4,3,3*(c%3+1)+1);plt.title(oLabel);plt.bar(xrange(len(data2)),data2);plt.xlim([0,100]);plt.ylim([-2,2]);
		plt.subplot(4,3,3*(c%3+1)+2);plt.title(oLabel);plt.bar(xrange(len(data3)),data3);plt.xlim([0,100]);plt.ylim([-2,2]);
		conv=newFindMatch2d(np.array(query),np.array([data2,data3]));
		dis.append(np.amin(conv));
		plt.subplot(4,3,3*(c%3+1)+3);plt.title("Variance Match");plt.bar(xrange(len(conv)),conv);plt.xlim([0,200]);plt.ylim([0,2]);
		plt.tight_layout();
	plt.tight_layout();
	plt.close();
	return(np.array(dis))


def newFindMatch2d(a,b):
	matchValues=[]
	if a.shape[1]<b.shape[1]:
		a,b=b,a;
	
	padding=np.zeros((2,b.shape[1]));
	a=np.hstack((padding,a,padding));
	for i in range(a.shape[1]):
		if i==a.shape[1]-b.shape[1]+1:
			break 
		x=a[:,i:i+b.shape[1]]
		matchValues.append(np.linalg.norm(np.std(x-b,1)))
	return np.array(matchValues);
	
def updateScore(score,labels,indexes,dis):
	for d,idx in zip(dis,indexes):
			score[idx]+=d;
	return score;
	
def findPrecisionRecall(qLabel,retLab):
	precision=np.zeros(len(retLab));
	recall=np.ones(len(retLab));
	cLabel=qLabel.split('_')[0];
	classes=[i.split('_')[0] for i in retLab];
	nMatch=0;
	recallDic=Counter(classes);
	if recallDic[cLabel]==0:
		return(precision,recall);
	else:
		
		for c1,label in enumerate(classes):
			if cLabel ==label:
				nMatch+=1.0;
			precision[c1]=nMatch/(c1+1);
			recall[c1]=nMatch/recallDic[cLabel]
		return(precision,recall);
		
