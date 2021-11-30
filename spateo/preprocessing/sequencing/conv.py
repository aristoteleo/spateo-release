
#!/usr/bin/env python

import numpy as np
import cv2
from scipy import signal

def conv(a, ks, circle=False):
	'''
	convolution with specific kernel size of all one kernel

	Parameters
    ----------
    a: a 2d np array
    	The source array.

	ks: an odd int
		The kernel size of kernel used to perform convolution. The elements of the kernel would be all ones.

	Returns
    -------
    c: a 2d np array
        The returned array has the same type as `a`.


    Examples
    --------
    >>> a = np.array([[1,2,3,4,5,6],[1,1,1,1,1,1],[1,300,1,1,1,1],[1,1,1,1,1,1]])
    >>> c = conv(a, ks=3, circle=True)
    >>> c
    array([[  6,   9,  13,  17,  21,  24],
       [  5, 305,   7,   8,   9,  10],
       [304, 304, 304,   5,   5,   5],
       [  5, 304,   5,   5,   5,   5]])

	'''

	# check if ks is an odd number
	if ks % 2 == 0:
		print("Please enter an odd int ks")
		return None
	kernel = np.ones([ks,ks], dtype=np.uint8)
	if circle:
		kernel = cv2.circle(np.zeros([ks,ks], dtype=np.uint8), (int((ks-1)/2),int((ks-1)/2)), int((ks-1)/2), 1, -1)
	c = signal.convolve2d(a, kernel, boundary='symm', mode='same')
	return c
