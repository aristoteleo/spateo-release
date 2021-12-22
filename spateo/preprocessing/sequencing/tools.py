
#!/usr/bin/env python

import numpy as np
import cv2
from scipy import signal
import time

def conv(a, ks, circle=False):
    '''
    convolution with specific kernel size of all one kernel

    Parameters
    ----------
    a: 2d np array
        The source array.

    ks: an odd int
        The kernel size of kernel used to perform convolution. The elements of the kernel would be all ones.

    Returns
    -------
    c: a 2d np array
    The returned array has the same dtype as `a`.


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



def array2img(array, outFig):
    '''
    Save a 2d numpy array as a 8 bits tif. Whatever the dtype of array is, it would be change to np.uint8 after
    truncating values larger than 255 to 255.
    :param array: input 2d numpy array
    :param outFig: str, output figure name
    :return: `array`
    '''
    a = array.copy()
    a[a>=255] = 255
    cv2.imwrite(outFig, a.astype(np.uint8))
    return(array)


def scaleTo255(x, inplace=True):
    '''
    scale 2d numpy array to 0.0 - 255.0
    :param x: 2d numpy array
    :param inplace: bool
    :return: None if `inplace=True` else 2d numpy array (0.0-255.0). If `inplace=True` would not change the dtype of `x`.
    '''
    x2 = scaleTo01(x)
    x2 *= 255
    if inplace:
        x[:] = x2
        return(None)
    else:
        return(x2)

def scaleTo01(array):
    y = (array-np.min(array)) / (np.max(array)-np.min(array))
    return y # 2d array 0.0 - 1.0

def gBlur(array, k, inplace=False):
    dst = cv2.GaussianBlur(src=array, ksize=(k, k), sigmaX=0.0, sigmaY=0.0)
    print(f"gBlur: min:{np.min(dst)} mean:{np.mean(dst)} median:{np.median(dst)} max:{np.max(dst)}")
    if inplace:
        array[:] = dst
        return None
    else:
        return dst


def getknee(gBlurArray): # np.float 0 - 255
    '''
    find the knee point of cumulative curve of gBlur results after change gBlur results to 0.0 - 255.0
    :param gBlurArray: 2d np array np.float64
    :return: the gBlur value of knee point, int
    '''
    allSpotNum = gBlurArray.shape[0] * gBlurArray.shape[1]

    x = []
    y = []
    #s = time.time()
    for i in range(1,250):
        tmp = len(gBlurArray[gBlurArray<=i])/allSpotNum
        if tmp>0.5:
            x.append(i)
            y.append(tmp)
    #e = time.time()
    #print(f'dict: {e-s}')


    from kneed import KneeLocator
    import matplotlib.pyplot as plt

    #s = time.time()
    kl = KneeLocator(x, y, curve="concave")
    kl.plot_knee()
    plt.savefig("knee.png")
    t = kl.knee
    print(f'cutoff: {t}')
    #e = time.time()
    #print(f'knee: {e-s}')
    return(round(kl.knee))


def addCellLabels(inFile, cellMask, x_min, y_min, outFile, cens):
    o = open(outFile, "wt")
    o.write("geneID\tx\ty\tUMICount\tlabel\tcentroid_y\tcentroid_x\n")
    with open(inFile, "rt") as f:
        f.readline()
        for line in f:
            lines = line.strip().split("\t")
            x, y = int(lines[0]), int(lines[1])
            x = x - x_min
            y = y - y_min
            if int(cellMask[y,x]) > 0:
                o.write(lines[2] + "\t" + lines[0] + "\t" + lines[1] + "\t" + lines[3] + "\t" + str(int(cellMask[y,x]))+"\t" + str(cens[int(cellMask[y,x])][0]+y_min) + "\t" + str(cens[int(cellMask[y,x])][1]+x_min) + "\n")
            else:
                o.write(lines[2] + "\t" + lines[0] + "\t" + lines[1] + "\t" + lines[3] + "\t0\t0\t0\n")
    o.close()
