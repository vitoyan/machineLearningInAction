from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) -1
    dataMat = []
    labMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labMat.append(float(curLine[-1]))
    return dataMat, labMat

def standRegres(xArr, yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def gradientDescent(xArr, yArr, alph, iterNum):
    xMat = mat(xArr)
    m,n = xMat.shape
    yMat = mat(yArr).T
    theta = ones((n, 1))
    theta = mat(theta)
    thetaHistory = []
    for iter in range(iterNum):
        theta = theta - alph*(xMat.T * (xMat*theta - yMat))/m
        thetaHistory.append(theta);
    return thetaHistory