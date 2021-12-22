#!/usr/bin/env python

import numpy as np
import sys
import os
import cv2
from scipy import signal, stats
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import pandas as pd
from . import bp, tools, watershed, enlarge, nbn, assign


class Icell:
    """
    a class for identifying cells by RNA signal.
    """

    def __init__(self, infile="", spliced=False, total=False):
        self.infile = infile
        self.spliced = spliced
        self.total = total
        self.xmin, self.xmax, self.ymin, self.ymax = (
            float("inf"),
            0,
            float("inf"),
            0,
        )
        self.rawdata = None  # 2d np array np.int16
        self.genenum = None  # 2d np array np.int16
        self.convdata = None  # 2d np array np.float64, storage convolution result, EM result, or BP result 0.0 - 255.0
        self.cellMask = None  # 2d np array np.uint8  0=>background 255=>cell
        self.cellLabels = (
            None  # 2d np array represents cell labels of the whole slice np.int32
        )
        # EM results. These are needed to use BP.
        self.em_n = None
        self.em_p = None

    def readData(self, record_genenum=False):
        if self.xmax == 0:
            self.getBor()
            print(
                f"xmin, xmax: {self.xmin, self.xmax}  ymin, ymax: {self.ymin, self.ymax}"
            )
        self.rawdata = np.zeros(
            [self.ymax - self.ymin + 1, self.xmax - self.xmin + 1],
            dtype=np.int16,
        )
        if record_genenum:
            self.genenum = np.zeros(
                [self.ymax - self.ymin + 1, self.xmax - self.xmin + 1],
                dtype=np.int16,
            )
        self.readFile2array(record_genenum)
        self.cellMask = np.zeros(
            [self.ymax - self.ymin + 1, self.xmax - self.xmin + 1],
            dtype=np.uint8,
        )
        self.comCellLabels = np.zeros(
            [self.ymax - self.ymin + 1, self.xmax - self.xmin + 1],
            dtype=np.int32,
        )

    def gBlurAndSaveToTif(
        self, ks=21, outFig="gblur.tif", logDeal=False, usenbnEM=False
    ):
        if not usenbnEM:
            if not logDeal:
                self.convdata = self.gBlur(
                    self.rawdata.astype(np.float), k=ks, inplace=False
                )
            if logDeal:
                self.convdata = self.gBlur(
                    np.log2((self.rawdata + 1).astype(np.float)),
                    k=ks,
                    inplace=False,
                )
        if usenbnEM:
            self.convdata = self.gBlur(self.convdata, k=ks, inplace=False)

        tools.scaleTo255(self.convdata)
        tools.array2img(self.convdata, outFig)

    def getKneeOfgBlur(self):
        cutoff = tools.getknee(self.convdata)
        return cutoff

    def maskCellFromgBlur(self, cutoff=8):
        self.cellMask = np.where(self.convdata >= cutoff, 255, 0).astype(np.uint8)
        self.saveCellMask2tif()

    def maskCellFromEM(self, cutoff=230):
        self.cellMask = np.where(self.convdata >= cutoff, 255, 0).astype(np.uint8)
        self.saveCellMask2tif()

    def mclose(self, kernelSize=5, outFig="adc.tif"):
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        self.cellMask = cv2.morphologyEx(self.cellMask, cv2.MORPH_CLOSE, kernel)
        self.saveCellMask2tif(outFig)

    def mopen(self, kernelSize=3, outFig="adco.tif"):
        kernel = np.zeros((kernelSize, kernelSize), np.uint8)
        kernel = cv2.circle(
            kernel,
            (int((kernelSize - 1) / 2), int((kernelSize - 1) / 2)),
            int((kernelSize - 1) / 2),
            1,
            -1,
        )
        self.cellMask = cv2.morphologyEx(self.cellMask, cv2.MORPH_OPEN, kernel)
        self.saveCellMask2tif(outFig)

    def getCellLabels(self, kernelgblur=21, min_distance=5):
        self.cellLabels = watershed.getCellLabels(
            self.cellMask,
            self.rawdata,
            kernelgblur=kernelgblur,
            min_distance=min_distance,
        )
        print(f"cell number: {np.max(self.cellLabels)}")
        self.drawCellLabels()

    def saveCellMask2tif(self, outFig="cellMask.tif"):
        tools.array2img(self.cellMask, outFig)

    def enlarge(self, kpixel=5, maxA=400):
        enlarge.enlarge(self.cellLabels, kpixel=kpixel, maxA=maxA)
        self.drawCellLabels(outFig="cellLabels.enlarge.tif")

    def drawCellLabels(self, outFig="cellLabels.tiff", colorful=False):
        watershed.drawCellLabels(self.cellLabels, outFig, colorful=colorful)

    """
    def filterCellLabelsByArea(self,cellLabels, minA=25):
        process.filterCellLabelsByArea(self.comCellLabels, minA=minA)
        t = [i for i in np.unique(self.comCellLabels)if i>0]
        cn = len(t)
        print(f'cell number after filtered: {cn}')
        self.drawCellLabels(outfig="cellLabels.filtered.tif")
    """

    def saveCellLabels(self, outfile="cellLabels.matrix"):
        if outfile == "cellLabels.matrix":
            outfile = f"cellLabels.xrange_{self.xmin}-{self.xmax}_yrange_{self.ymin}-{self.ymax}.matrix"
        np.savetxt(outfile, self.cellLabels, fmt="%s")

    def addCellLabelsToMatrix(self, outFile):
        cens = assign.cal_cell_centriod(self.cellLabels)
        tools.addCellLabels(
            self.infile, self.cellLabels, self.xmin, self.ymin, outFile, cens
        )

    def assignNonCellSpots(
        self, long_matrix_file="expressionCellLabels.matrix", radius=1
    ):
        cens = assign.cal_cell_centriod(self.cellLabels)
        assign.assign_point(
            long_matrix_file,
            self.cellLabels,
            cens,
            self.xmin,
            self.ymin,
            radius=radius,
        )

    # nbnEM
    def nbnEM(
        self,
        k=11,
        w=np.array([0.99, 0.01]),
        mu=np.array([10.0, 300.0]),
        var=np.array([20.0, 400.0]),
        maxitem=2000,
        precision=1e-3,
        usePeaks=False,
        tissueM=None,
    ):
        # b = cv2.circle(np.zeros([k,k], dtype=np.int16), (int((k-1)/2),int((k-1)/2)), int((k-1)/2), 1, -1)
        # c = signal.convolve2d(self.rawdata, b, boundary='symm', mode='same')
        c = tools.conv(self.rawdata, ks=k, circle=True)
        if usePeaks:
            # b = cv2.circle(np.zeros([k,k], dtype=np.int16), (int((k-1)/2),int((k-1)/2)), int((k-1)/2), 1, -1)
            # c = signal.convolve2d(self.rawdata, b, boundary='symm', mode='same')
            peaks, labels = self.getpickclean(c, min_distance=k)
            # print(peaks)
            peaks = peaks - 1
            self.drawHist(peaks)
            w, lam, theta = nbn.nbnEM(
                peaks,
                c,
                w=w,
                mu=mu,
                var=var,
                maxitem=maxitem,
                precision=precision,
            )
        else:
            if isinstance(tissueM, np.ndarray):
                peaks = c[tissueM == 0]
            else:
                peaks = c.flatten()
            if len(peaks) > 1000000:
                peaks = np.random.choice(peaks, 1000000)
            self.drawHist(peaks)
            w, lam, theta = nbn.nbnEM(
                peaks,
                c,
                w=w,
                mu=mu,
                var=var,
                maxitem=maxitem,
                precision=precision,
            )

        self.em_n = -lam / np.log(theta)
        self.em_p = theta
        posprob = nbn.posp(w, lam, theta, c)
        self.convdata = posprob
        print(f"max posprob: {np.max(posprob)}")
        tools.scaleTo255(posprob, inplace=True)
        tools.array2img(posprob, outFig="nbnEM.tif")

    def bp(
        self,
        neighborhood=None,
        p=0.7,
        q=0.3,
        precision=1e-3,
        max_iter=100,
        n_threads=1,
    ):
        if self.em_n is None or self.em_p is None:
            raise Exception("Run EM first")

        background_probs = stats.nbinom(n=self.em_n[0], p=self.em_p[0]).pmf(
            self.rawdata
        )
        cell_probs = stats.nbinom(n=self.em_n[1], p=self.em_p[1]).pmf(self.rawdata)
        posprob = bp.cell_marginals(
            cell_probs,
            background_probs,
            neighborhood=neighborhood,
            p=p,
            q=q,
            precision=precision,
            max_iter=max_iter,
            n_threads=n_threads,
        )
        self.convdata = posprob
        tools.scaleTo255(posprob, inplace=True)
        tools.array2img(posprob, outFig="BP.tif")

    def drawHist(self, a, zero2one=False):
        if zero2one:
            plt.hist(a, bins=np.arange(0, 1, 0.01))
        else:
            # plt.hist(a, bins=range(0,int(np.max(a))+1))
            plt.hist(a, bins=100, range=[0, 100])
            plt.savefig("convDis.png")

    def getpickclean(self, img, min_distance=21):
        picks = peak_local_max(img, min_distance=min_distance)
        b = np.zeros(img.shape, dtype=np.uint8)
        b[picks[:, 0], picks[:, 1]] = 1
        num_objects, labels = cv2.connectedComponents(b)
        rs = []
        tmp = {}
        for y in range(len(labels)):
            for x in range(
                len(
                    labels[
                        y,
                    ]
                )
            ):
                if labels[y, x] > 0 and labels[y, x] not in tmp:
                    tmp[labels[y, x]] = 1
                    rs.append(img[y, x])
        return (np.array(rs), labels)

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

    # self.rawdata[0,0] => the value in (self.ymin,self.xmin)
    # self.rawdata[0,1] => the value in (self.ymin,self.xmin + 1)
    def readFile2array(self, record_genenum):
        a = [
            [0] * (self.xmax - self.xmin + 1)
            for _ in range((self.ymax - self.ymin + 1))
        ]
        g = (
            [
                [0] * (self.xmax - self.xmin + 1)
                for _ in range((self.ymax - self.ymin + 1))
            ]
            if record_genenum
            else 0
        )
        with open(self.infile, "rt") as f:
            f.readline()
            for line in f:
                lines = line.strip().split("\t")
                if int(lines[3]) <= 0:
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

                if record_genenum:
                    g[y][x] += 1
        self.rawdata = np.array(a, dtype=np.int16)
        self.genenum = np.array(g, dtype=np.int16) if record_genenum else None

    def saveRawAstif(self, outFig="raw.tif"):
        tools.array2img(self.rawdata, outFig=outFig)

    def scaleTo255(self, x):
        process.scaleTo255(x)

    def gBlur(self, x, k, inplace=False):
        dst = tools.gBlur(x, k, inplace)
        if not inplace:
            return dst
