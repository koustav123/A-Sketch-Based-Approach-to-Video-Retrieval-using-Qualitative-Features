import numpy as np
import pickle
class Parameters:

	def __init__(self):
		#self.clusterModelPath="/home/koustav/Desktop/multimodality/codes/scripts/Python/results/clusterModel.p"
		#self.codebookPath="/home/koustav/Desktop/multimodality/codes/scripts/Python/results/Codebook.txt"
		
		self.svmModelPath="/home/koustav/Desktop/multimodality/codes/scripts/Python2/Pool/results/svm_model.p"
		self.model=pickle.load(open(self.svmModelPath,"rb"));
		
		#self.kmModelPath="/home/koustav/Desktop/multimodality/codes/scripts/Python2/Pool/results/kmeansModel.p";
		#self.model=pickle.load(open(self.kmModelPath,"rb"));
		
		self.xLimit=200;
		self.yLimit=110;
		self.segmentationAngle=165;
		self.splSm=.5; # spline smoothness factor
		self.splUp=1.0#upsampling for spline
		self.splDown=0.25#downsampling for spline
		self.splineOrder=1;
		self.radC=0.012
		#self.motionNature=[1,0.8,0.6,0.4,0.2]
		#self.motionNature=[1,0.6,0.3]
		self.featureLength=4;
		self.levelOneFeatureLength=28;
		self.eqSeg=5;
		self.minSegLen=4; #the number of points in the segments that should atleast be present for interpolation
		self.eqP=20; #upsampling factor during equisampling
		self.eqSegT=4; #distance between points threshold
		self.nDir=range(-180,180,45);
		self.cleanThreshold=5 # threshold in cleanTrajectory() to remove noise
		self.outlierThreshold=200;#threshold for outlier removal
		self.distributionThreshold=50 # threshold for quadrants
		self.nn=200 # number of nearest neighbours for level one filter
		self.l1c=200;
		self.l2c=200; # number of results after level 2
		self.l3c=200;
		self.l4c=200;
		self.l4_5c=200;
		self.l5c=200;
		self.topk=15;
		self.videoFrameSize=[425,200];
		self.userFrameSize=[1250,565];
		self.salStrokeSize=20;
		
