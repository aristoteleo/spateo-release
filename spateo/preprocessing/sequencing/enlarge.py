#!/usr/bin/env python

import numpy as np
import cv2
import random
import time

def enlarge(a, kpixel=5, maxA=400):
	for i in range(kpixel):
		aexp = np.zeros([a.shape[0]+2, a.shape[1]+2], dtype=np.int32)
		aexp[1:-1,1:-1] = a
		top = aexp[0:-2,1:-1]
		left = aexp[1:-1,0:-2]
		right = aexp[1:-1,2:]
		bottom = aexp[2:,1:-1]

		allCellLabels = np.unique(a)
		allCellLabels = [c for c in allCellLabels if c>0]
		#cWithmaxA = [c for c in allCellLabels if (a==c).sum() >= maxA]
		cellArea = np.bincount(a.flatten())
		cWithmaxA = np.argwhere(cellArea>=maxA)
		cWithmaxA = cWithmaxA[cWithmaxA>0]

		top[np.isin(top, cWithmaxA)] = 0
		left[np.isin(left, cWithmaxA)] = 0
		right[np.isin(right, cWithmaxA)] = 0
		bottom[np.isin(bottom, cWithmaxA)] = 0

		a[(a==0) & (top>0)] = top[(a==0) & (top>0)]
		a[(a==0) & (left>0)] = left[(a==0) & (left>0)]
		a[(a==0) & (right>0)] = right[(a==0) & (right>0)]
		a[(a==0) & (bottom>0)] = bottom[(a==0) & (bottom>0)]
		#print(a)
		cv2.imwrite(f"{i}.tif", a)

def main():
	#a = cv2.imread("cellLabels.tiff",0).astype(np.int32)
	#print(a.dtype)
	a = np.array([[0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0],
				  [0,0,1,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,2,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,3,3,0],
				  [0,0,0,0,0,0,0,0,0,0]], dtype=np.int32)
	print(a)
	enlarge(a,5,6)
	print(a)

if __name__ == "__main__":
	main()
