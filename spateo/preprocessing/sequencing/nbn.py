#!/usr/bin/env python

import numpy as np
from scipy.stats import nbinom
from scipy.special import digamma

def nbnEMOne(w, lam, theta, x):
    #nbinom.pmf(k, n, p) k:x n:r    p:theta         in paper
    r = -lam/np.log(theta)
    bp = nbinom.pmf(k=x, n=r[0], p=theta[0])
    cp = nbinom.pmf(k=x, n=r[1], p=theta[1])
    tau = []
    tau.append(w[0] * bp)
    tau.append(w[1] * cp)
    tau = np.array(tau)
    mu = lamtheta2muvar(lam, theta)[0]
    tau[0,(tau.sum(axis=0)<=1e-9) & (x<mu[0]*2)] = 1
    tau[1,(tau.sum(axis=0)<=1e-9) & (x>=mu[0]*2)] = 1
    tau = tau/tau.sum(axis=0)

    beta = 1-1/(1-theta)-1/np.log(theta)

    delta = []
    delta.append(r[0]*(digamma(r[0]+x)-digamma(r[0])))
    delta.append(r[1]*(digamma(r[1]+x)-digamma(r[1])))
    delta = np.array(delta)


    tmp = tau.sum(axis=1)
    w[:] = tmp/tmp.sum()
    lam[:] = (tau*delta).sum(axis=1)/tau.sum(axis=1)
    thetatmp = []
    thetatmp.append(beta[0]*(tau[0]*delta[0]).sum()/(tau[0]*(x-(1-beta[0])*delta[0])).sum())
    thetatmp.append(beta[1]*(tau[1]*delta[1]).sum()/(tau[1]*(x-(1-beta[1])*delta[1])).sum())
    theta[:] = np.array(thetatmp)

def posp(w, lam, theta, x):
    r = -lam/np.log(theta)
    bp = nbinom.pmf(k=x, n=r[0], p=theta[0])
    cp = nbinom.pmf(k=x, n=r[1], p=theta[1])
    tau = []
    tau.append(w[0] * bp)
    tau.append(w[1] * cp)
    tau = np.array(tau)
    tau = tau/tau.sum(axis=0)
    return(tau[1]) # the prob of cell

def lam2r(lam, theta):
    r = -lam/np.log(theta)
    return(r)

def muvar2lamtheta(mu, var):
    r = mu**2/(var-mu)
    theta = mu/var
    lam = -r*np.log(theta)
    return(lam, theta)

def lamtheta2muvar(lam, theta):
    r = lam2r(lam, theta)
    mu = r/theta - r
    var = mu + mu**2/r
    return(mu, var)


def nbnEM(x, realData, w=np.array([0.99,0.01]), mu=np.array([10.0,100.0]), var=np.array([20.0,200.0]), maxitem=2000, precision=1e-3):
    maxitem = maxitem
    precision = precision
    w = w # bacground cell
    lam, theta = muvar2lamtheta(mu, var)

    for i in range(0, maxitem):
        wpre = w.copy()
        lampre = lam.copy()
        thetapre = theta.copy()
        nbnEMOne(w, lam, theta, x)
        print(f"{i} item")
        print(f"w: {w}")
        print(f"lam: {lam}")
        print(f"r: {lam2r(lam, theta)}")
        print(f"theta: {theta}")
        print(f"mu: {lamtheta2muvar(lam, theta)[0]}")
        print(f"var: {lamtheta2muvar(lam, theta)[1]}")
        if np.max([np.max(np.fabs(w-wpre)), np.max(np.fabs(lam-lampre)), np.max(np.fabs(theta-thetapre))]) <= precision:
            print(f"break in {i} item")
            break
    return w, lam, theta

def readData(infile):
    data = []
    with open(infile, "rt") as f:
        for line in f:
            line = int(line.strip())
            data.append(line)
    return(np.array(data))

#x = readData("sim11.txt") # 1d
#posprob = nbnEM(x, realData) # 2d
