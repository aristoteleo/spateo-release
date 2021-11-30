#!/usr/bin/env python

import numpy as np
import sys
import os
from . import process
from . import watershed
import cv2
from icell import uti
from functools import wraps
from . import enlarge
from icell import nbn
from scipy import signal
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import pandas as pd


class Icell:
	'''
	a class for identifying cells by RNA signal.
	'''
	def __init__(self, infile="", spliced=False, total=False):
		self.infile = infile
		self.subwidth = 0
		self.subheight = 0
		self.spliced = spliced
		self.total = total
		self.xmin, self.xmax, self.ymin, self.ymax = float('inf'), 0, float('inf'), 0
		self.rawdata = None # np array np.int16
		self.genenum = None # np array np.int16
		self.convdata = None # np array np.float64
		self.splited = False
		self.cellMask = None # np array np.uint8  0=>background 255=>cell
		self.cellLabels = None # a np array represents cell labels of the whole slice np.int32
	
	
	def readData(self):
		if self.xmax == 0:
			self.getBor()
			print(f'xmin, xmax: {self.xmin, self.xmax}  ymin, ymax: {self.ymin, self.ymax}')
		self.rawdata = np.zeros([self.ymax-self.ymin+1, self.xmax-self.xmin+1], dtype=np.int16)
		self.genenum = np.zeros([self.ymax-self.ymin+1, self.xmax-self.xmin+1], dtype=np.int16)
		self.readFile2array()
		self.cellMask = np.zeros([self.ymax-self.ymin+1, self.xmax-self.xmin+1], dtype=np.uint8)
		self.comCellLabels = np.zeros([self.ymax-self.ymin+1, self.xmax-self.xmin+1], dtype=np.int32)
	
	'''
	def readDataNew(self):
		df = pd.read_table(self.infile)
		if self.total:
			a = pd.pivot_table(df, values='MIDCounts', index='y', columns='x', fill_value=0, aggfunc=np.sum)
			b = pd.pivot_table(df, values='MIDCounts', index='y', columns='x', fill_value=0, aggfunc=np.count_nonzero)
		if self.spliced:
			a = pd.pivot_table(df, values='EXONIC', index='y', columns='x', fill_value=0, aggfunc=np.sum)
			b = pd.pivot_table(df, values='EXONIC', index='y', columns='x', fill_value=0, aggfunc=np.count_nonzero)
		if not self.spliced and not self.total:
			a = pd.pivot_table(df, values='INTRONIC', index='y', columns='x', fill_value=0, aggfunc=np.sum)
			b = pd.pivot_table(df, values='INTRONIC', index='y', columns='x', fill_value=0, aggfunc=np.count_nonzero)
		if self.spliced and self.total:
			print("spliced and total are mutually exclusive, please set only one of them")
			sys.exit("sorry, goodbye!")

		self.rawdata = np.array(a, dtype=np.int16)
		self.genenum = np.array(b, dtype=np.int16)

		self.xmin = a.index[0]
		self.xmax = a.index[-1]
		self.ymin = a.columns.values[0]
		self.ymax = a.columns.values[-1]
	
		self.cellMask = np.array(self.rawdata.shape, dtype=np.uint8)
		self.comCellLabels = np.array(self.rawdata.shape, dtype=np.int32)
	'''
	
	
	def saveObj(self, out="ins.pkl"):
		import pickle
		f = open(out, "wb")
		f.write(pickle.dumps(self))
		f.close()

	def loadObj(self, inpkl="ins.pkl"):
		import pickle
		f = open(inpkl, "rb")
		self.__dict__.update(pickle.loads(f.read()).__dict__)
		print("complete loading obj")

	def selectSpotsWithinCell(self, blockSize=21, C=2, outfig="ad.tif"):
		# spots within a cell and have nearly zero value would be white (255). Others would be black (0)
		self.rawdata[self.rawdata>255] = 255
		self.rawdata = self.rawdata.astype(np.uint8)
		self.cellMask = cv2.adaptiveThreshold(self.rawdata,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,C)
		self.saveCellMask2tif(outfig)
	
	def getCellMaskWithKmeans(self, ks=21, outfig="cellMask.tif", c1toc2Ratio=2.0, cellRatio=0.3, lg=False):
		# spot belongs to a cell would be 255. Others would be 0
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,50,1)
		flags = cv2.KMEANS_PP_CENTERS
		data1 = self.gBlur(self.rawdata.astype(np.float), ks, inplace=False)
		data2 = self.gBlur(self.genenum.astype(np.float), ks, inplace=False)
		data1 = data1.reshape((data1.shape[0]*data1.shape[1],1))
		data2 = data2.reshape((data2.shape[0]*data2.shape[1],1))
		datac = np.hstack((data1, data2))
		del data1,data2
		datac = np.float32(datac)
		if lg:
			datac = np.log(datac+0.001)
		compactness,labels,centers = cv2.kmeans(datac,3,None,criteria,10,flags)
		print(f'centers:\n{centers}')
		self.cellMask = labels.reshape((self.rawdata.shape[0],self.rawdata.shape[1]))
		self.cellMask = self.cellMask.astype(np.uint8)
		self.cellMask *= 100
		self.saveCellMask2tif(outfig='kmeanCluster.tif')
		self.cellMask = self.cellMask/100
		self.cellMask = self.cellMask.astype(np.uint8)
		#cenidx = np.argsort(centers, axis=0)
		#cenidx = cenidx[:,0]
		cenidx = []
		for idx in range(len(centers[:,0])):
			cenidx.append([centers[:,0][idx],idx])
		cenidx.sort(key=lambda x:x[0])
		cenidx = np.array(cenidx)[:,1]

		for i in range(len(cenidx)):
			self.cellMask[self.cellMask==cenidx[i]] = i + 3
		self.cellMask -= 3
		edgeToCellRatio = (self.cellMask==1).sum()/(self.cellMask==2).sum()
		cellDensity = ((self.cellMask==1).sum() + (self.cellMask==2).sum())/(self.cellMask.shape[0]*self.cellMask.shape[1])
		print(f'edgeToCellRatio: {edgeToCellRatio}')
		print(f'cellDensity in term of considering both cluster 1 and cluster 2: {cellDensity}')
		if edgeToCellRatio <= c1toc2Ratio and cellDensity <= cellRatio:
			self.cellMask[self.cellMask==1] = 2

		self.cellMask[self.cellMask<2] = 0
		self.cellMask[self.cellMask==2] = 255
		self.saveCellMask2tif(outfig)

	def gBlurSubtractRaw(self, ks=21, outfig="gblur-raw.tif"):
		self.convdata = self.gBlur(self.rawdata.astype(np.float), k=ks, inplace=False)
		self.convdata -= self.rawdata
		self.convdata[self.convdata<0] = 0
		process.scaleTo255(self.convdata)
		process.array2img(self.convdata, outfig)

	def gBlurAndSaveToTif(self, ks=21, outfig="gblur.tif", logDeal=False, usenbnEM=False):
		if not usenbnEM:
			if not logDeal:
				self.convdata = self.gBlur(self.rawdata.astype(np.float), k=ks, inplace=False)
			if logDeal:
				self.convdata = self.gBlur(np.log2((self.rawdata+1).astype(np.float)), k=ks, inplace=False)
		if usenbnEM:
			self.convdata = self.gBlur(self.convdata, k=ks, inplace=False)

		process.scaleTo255(self.convdata)
		process.array2img(self.convdata, outfig)

	def getKneeOfgBlur(self):
		cutoff = process.getknee(self.convdata)
		return cutoff
	
	def maskCellFromgBlur(self, cutoff=8):
		self.cellMask = np.where(self.convdata>=cutoff, 255, 0).astype(np.uint8)
		self.saveCellMask2tif()

	def	maskCellFromEM(self, cutoff=230):
		self.cellMask = np.where(self.convdata>=cutoff, 255, 0).astype(np.uint8)
		self.saveCellMask2tif()

	def mclose(self, kernelSize=5, outfig="adc.tif"):
		kernel = np.ones((kernelSize, kernelSize), np.uint8)
		self.cellMask = cv2.morphologyEx(self.cellMask, cv2.MORPH_CLOSE, kernel)
		self.saveCellMask2tif(outfig)
	
	def mopen(self, kernelSize=3, outfig="adco.tif"):
		kernel = np.zeros((kernelSize, kernelSize), np.uint8)
		kernel = cv2.circle(kernel,(int((kernelSize-1)/2),int((kernelSize-1)/2)),int((kernelSize-1)/2),1,-1)
		self.cellMask = cv2.morphologyEx(self.cellMask, cv2.MORPH_OPEN, kernel)
		self.saveCellMask2tif(outfig)

	def getCellLabels(self, kernelgblur=21, min_distance=5):
		self.comCellLabels = watershed.getCellLabels(self.cellMask, self.rawdata, kernelgblur=kernelgblur, min_distance=min_distance)
		print(f'cell number: {np.max(self.comCellLabels)}')
		self.drawCellLabels()

	def saveCellMask2tif(self, outfig="cellMask.tif"):
		cv2.imwrite(outfig, self.cellMask)

	def enlarge(self, kpixel=5, maxA=400):
		enlarge.enlarge(self.comCellLabels, kpixel=kpixel, maxA=maxA)
		self.drawCellLabels(outfig="cellLabels.enlarge.tif")

	def drawCellLabels(self, outfig="cellLabels.tiff", colorful=False):
		watershed.drawCellLabels(self.comCellLabels, outfig, colorful=colorful)

	def filterCellLabelsByArea(self,cellLabels, minA=25):
		process.filterCellLabelsByArea(self.comCellLabels, minA=minA)
		t = [i for i in np.unique(self.comCellLabels)if i>0]
		cn = len(t)
		print(f'cell number after filtered: {cn}')
		self.drawCellLabels(outfig="cellLabels.filtered.tif")

	def saveCellLabels(self, outfile='cellLabels.matrix'):
		if outfile == 'cellLabels.matrix':
			outfile=f'cellLabels.xrange_{self.xmin}-{self.xmax}_yrange_{self.ymin}-{self.ymax}.matrix'
		np.savetxt(outfile, self.comCellLabels, fmt='%s')

	def addCellLabelsToMatrix(self, outFile):
		uti.addCellLabels(self.infile, self.comCellLabels, self.xmin, self.ymin, outFile)

	# nbnEM
	def nbnEM(self, k=11, w=np.array([0.99,0.01]), mu=np.array([10.0,300.0]), var=np.array([20.0,400.0]), maxitem=2000, precision=1e-3, usePeaks=True, tissueM=None):
		b = cv2.circle(np.zeros([k,k], dtype=np.int16), (int((k-1)/2),int((k-1)/2)), int((k-1)/2), 1, -1)
		c = signal.convolve2d(self.rawdata, b, boundary='symm', mode='same')
		if usePeaks:
			#b = cv2.circle(np.zeros([k,k], dtype=np.int16), (int((k-1)/2),int((k-1)/2)), int((k-1)/2), 1, -1)	
			#c = signal.convolve2d(self.rawdata, b, boundary='symm', mode='same')
			peaks, labels = self.getpickclean(c, min_distance=k)
			#print(peaks)
			peaks = peaks - 1
			self.drawHist(peaks)
			posprob = nbn.nbnEM(peaks, c, w=w, mu=mu, var=var, maxitem=maxitem, precision=precision)
		else:
			if isinstance(tissueM, np.ndarray):
				peaks = c[tissueM==0]
			else:
				peaks = c.flatten()
			peaks = np.random.choice(peaks,1000000)
			self.drawHist(peaks)
			posprob = nbn.nbnEM(peaks, c, w=w, mu=mu, var=var, maxitem=maxitem, precision=precision)

		self.convdata = posprob
		print(f'max posprob: {np.max(posprob)}')
		process.scaleTo255(posprob)
		process.array2img(posprob, outFig="nbnEM.tif")

	def drawHist(self, a, zero2one=False):
		if zero2one:
			plt.hist(a, bins=np.arange(0,1,0.01))
		else:
			#plt.hist(a, bins=range(0,int(np.max(a))+1))
			plt.hist(a, bins=100,range=[0,100])
			plt.savefig("peaks.png")
		
	def getpickclean(self, img, min_distance=21):
		picks = peak_local_max(img,min_distance=min_distance)
		b = np.zeros(img.shape, dtype=np.uint8)
		b[picks[:,0],picks[:,1]] = 1
		num_objects, labels = cv2.connectedComponents(b)
		rs = []
		tmp = {}
		for y in range(len(labels)):
			for x in range(len(labels[y,])):
				if labels[y,x]>0 and labels[y,x] not in tmp:
					tmp[labels[y,x]] = 1
					rs.append(img[y,x])
		return(np.array(rs), labels)
		
		
	

	# the spot coordinate starts from 0
	def getBor(self):
		with open(self.infile, "rt") as f:
			f.readline()
			for line in f:
				lines = line.strip().split("\t")
				x, y = int(lines[0]), int(lines[1])
				self.xmin = x if x < self.xmin else self.xmin
				self.xmax = x if x > self.xmax else self.xmax

				self.ymin = y if y < self.ymin else self.ymin
				self.ymax = y if y > self.ymax else self.ymax

	
	# self.data[0,0] => the value in (self.ymin,self.xmin)
	# self.data[0,1] => the value in (self.ymin,self.xmin + 1)
	def readFile2array(self):
		a = [[0]*(self.xmax-self.xmin+1) for _ in range((self.ymax-self.ymin+1))]
		g = [[0]*(self.xmax-self.xmin+1) for _ in range((self.ymax-self.ymin+1))]
		with open(self.infile, "rt") as f:
			f.readline()
			for line in f:
				lines = line.strip().split("\t")
				if int(lines[3])<=0:
					continue
				x, y = int(lines[0]), int(lines[1])
				x = x - self.xmin
				y = y - self.ymin
				if self.total:
					a[y][x] += int(lines[3])
				elif self.spliced:
					a[y][x] += int(lines[4])
				else:
					a[y][x] += int(lines[5])
				
				g[y][x] += 1
		self.rawdata = np.array(a, dtype=np.int16)
		self.genenum = np.array(g, dtype=np.int16)
	



	def save2tif(self, x, des=""):
		process.array2img(x, des)

	def scaleTo255(self, x):
		process.scaleTo255(x)

	def scaleTo255int(self, x):
		process.scaleTo255int(x)

	def scaleTo10int(self, x):
		process.scaleTo10int(x)

	def scaleTo100int(self, x):
		process.scaleTo100int(x)

	def scaleTo200int(self, x):
		process.scaleTo200int(x)

	
	def scaleTo01(self, x):
		process.scaleTo01(x)

	def con2d3by3(self, x):
		process.con2d3by3(x)

	def gBlur(self, x, k, inplace=True):
		dst = process.gBlur(x, k, inplace)
		if not inplace:
			return(dst)
			

	def con2d(self, x, k):
		process.con2d(x, k)


	   
	def con2dC(self, x, k):
		process.con2dC(x, k)

	def opening(self, array, cutoff, ks):
		process.opening(array, cutoff, ks)


	def enlargeCellEdge(self, x, pixels, maxpixels):
		enlarge.expand(x, pixels, maxpixels)


	def addCellMask(self, outFile):
		if self.splited:
			uti.addCellMask(self.infile, self.comCellLabels, self.xmin, self.ymin, outFile)
		else:
			uti.addCellMask(self.infile, self.cellLabels, self.xmin, self.ymin, outFile)

	


