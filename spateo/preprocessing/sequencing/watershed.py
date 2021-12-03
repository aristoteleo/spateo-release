#!/use/bin/env python

from scipy import ndimage as ndi
from skimage import morphology,feature
from . import process
import numpy as np
import cv2
from skimage import color

def getCellLabels(cellmask, rawdata, kernelgblur=21, min_distance=5): # cellmask is a np array np.uint8 0=>background 255=>cell
	gblur = process.gBlur(rawdata.astype(float), kernelgblur, inplace=False)
	#print(gblur.dtype)
	#print(gblur)
	process.scaleTo255(gblur)
	#print(gblur.dtype)
	#print(gblur)
	#np.savetxt("gblur.txt",gblur)
	#cv2.imwrite("gblur.tif", gblur.astype(np.uint8))
	gblur[cellmask==0] = 0
	local_maxi = feature.peak_local_max(image=gblur, min_distance=min_distance, indices=False, labels=cellmask) # find peak
	markers = ndi.label(local_maxi)[0]
	labels = morphology.watershed(-gblur, markers, mask=cellmask)
	return(labels) # np array np.int32


def drawCellLabels(labels, outFig="cellLabels.tiff", colorful=False):
	if colorful == False:
		labels2 = labels.copy()
		labels2 %= 4 # np array np.int32
		labels2[labels2==0] = 100
		labels2[labels2==1] = 150
		labels2[labels2==2] = 200
		labels2[labels2==3] = 255
		labels2[labels==0] = 0
		cv2.imwrite(outFig, labels2.astype(np.uint8))

	else:
		dst = color.label2rgb(labels, bg_label=0)
		dst = dst * 255
		cv2.imwrite("cellLabels.color.tiff", dst.astype(np.uint8))
	
