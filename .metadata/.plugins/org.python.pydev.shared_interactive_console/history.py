uAtX.shape
uAtX
concatenate((essentialBCAt0,uAtX,))
np.concatenate((essentialBCAt0,uAtX,))
a = np.concatenate((essentialBCAt0, uAtX, essentialBCAt1), axis=0)
exampleXCoord
exampleXCoord.size
len(exampleXCoord)
uAtX
uAtX.tolist()
essentialBCAt0 + uAtX.tolist()
list(essentialBCAt0) + uAtX.tolist()
a = list()
a.append(essentialBCAt0)
a
a + uAtX.tolist()
a.append(essentialBCAt1)
a
a
a + uAtX.tolist()
a += uAtX.tolist()
a.append(essentialBCAt1)
a
len(a)
a = list()
a.append(essentialBCAt0)
a += uAtX
a.append(essentialBCAt1)
plt.plot(exampleXCoord, a)
exampleXCoord
a
a = list()
a.append(essentialBCAt0)
a += uAtX.tolist()
a.append(essentialBCAt1)
a
plt.plot(exampleXCoord, a)
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
plt.plot(exampleXCoord, uAtXAnalytic)
uAtX_concat = list()
uAtX_concat.append(essentialBCAt0)
uAtX_concat += uAtX.tolist()
uAtX_concat.append(essentialBCAt1)
plt.figure()
fdPlotHandle = plt.plot(exampleXCoord, uAtX_concat, label='fdPlot')
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
analyticPlotHandle = plt.plot(exampleXCoord, uAtXAnalytic, label='analyticPlot')
plt.legend = (fdPlotHandle, analyticPlotHandle)
plt.figure()
fdPlotHandle = plt.plot(exampleXCoord, uAtX_concat, label='fdPlot')
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
analyticPlotHandle = plt.plot(exampleXCoord, uAtXAnalytic, label='analyticPlot')
plt.legend = ([fdPlotHandle, analyticPlotHandle], ['fd', 'analytic'])
plt.show()
legend = plt.legend([fdPlotHandle, analyticPlotHandle], ['fd', 'analytic'])
plt.figure()
fdPlotHandle = plt.plot(exampleXCoord, uAtX_concat, label='fdPlot')
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
analyticPlotHandle = plt.plot(exampleXCoord, uAtXAnalytic, label='analyticPlot')
legend = plt.legend()#[fdPlotHandle, analyticPlotHandle], ['fd', 'analytic'])
legend = plt.legend()#[fdPlotHandle, analyticPlotHandle], ['fd', 'analytic'])
plt.legend()#[fdPlotHandle, analyticPlotHandle], ['fd', 'analytic'])
plt.show()
plt.legend(loc='upper right', shadow=True, fontsize='x-large')#[fdPlotHandle, analyticPlotHandle], ['fd', 'analytic'])
plt.legend([fdPlotHandle, analyticPlotHandle],['fd', 'analytic'])
plt.legend()
import matplotlib
matplotlib.__version__
plt.legend(handles = [fdPlotHandle, analyticPlotHandle])
plt.figure()
fdPlotHandle = plt.plot(exampleXCoord, uAtX_concat, label='fdPlot')
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
analyticPlotHandle = plt.plot(exampleXCoord, uAtXAnalytic, label='analyticPlot')
plt.legend(loc='upper right')#handles = [fdPlotHandle, analyticPlotHandle])
plt.legend
plt.show()
fig = plt.figure()
fdPlotHandle = plt.plot(exampleXCoord, uAtX_concat, label='fdPlot')
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
analyticPlotHandle = plt.plot(exampleXCoord, uAtXAnalytic, label='analyticPlot')
fig.legend()
fig.legend((fdPlotHandle,analyticPlotHandle), ('fd', 'analytic'), 'best')
fig.show()
plt.show()
fig.legend((fdPlotHandle,analyticPlotHandle), ('fd', 'analytic'), 'best')
plt.legend(loc='upper right')#handles = [fdPlotHandle, analyticPlotHandle])
fig.legend((fdPlotHandle,analyticPlotHandle), ('fd', 'analytic'), 'best')
plt.legend(loc='upper right')#handles = [fdPlotHandle, analyticPlotHandle])
plt.legend(loc='upper right')#handles = [fdPlotHandle, analyticPlotHandle])
plt.legend(handles = [fdPlotHandle, analyticPlotHandle])
plt.legend()#handles = [fdPlotHandle, analyticPlotHandle])
import sys; print('%s %s' % (sys.executable or sys.platform, sys.version))
import sys; print('%s %s' % (sys.executable or sys.platform, sys.version))
import sys; print('%s %s' % (sys.executable or sys.platform, sys.version))
import sys; print('%s %s' % (sys.executable or sys.platform, sys.version))
import numpy as np
from matplotlib import pyplot as plt
import random
def solveBySpectralMethod(K, f):
    """Step 1
    w -> eigenvalues of K
    v -> eigenvectors of K"""
    w,v = np.linalg.eig(K)
    print "Dot product of column 0 and column 1: ", np.dot(v[:,0], v[:,1])
    print "Dot product of column 1 and column 2: ", np.dot(v[:,1], v[:,2])
    print "Dot product of column 0 and column 2: ", np.dot(v[:,0], v[:,2])
    print "Inner product of column 0 and column 1: ", innerproduct(v[:,0], v[:,1])
    print "Inner product of column 1 and column 2: ", innerproduct(v[:,1], v[:,2])
    print "Inner product of column 0 and column 2: ", innerproduct(v[:,0], v[:,2])
    print 'Checking orthogonality of eigenvectors...\n'
    c_i = 0.0
    for i in range(v.shape[1]):
        c_i += np.dot(v[:,i], w)/(np.dot(v[:,i], v[:,i]) * w)
    u = sum(c_i * v)
    return u
def innerproduct(vector1, vector2):
    return sum(vector1 * vector2)
nE1 = 4.0
domainX = [0.0, 1.0]
essentialBCAt0 = 0.0
essentialBCAt1 = 0.0
hElement = (domainX[1] - domainX[0])/nE1
i = 0
exampleXCoord = list()
while i <= 1:
    exampleXCoord.append(i)
    i += hElement
nNodes = nE1 - 1
fdEqnMaskI = [-2, 1]
fdEqnMaskJ = [1, -2, 1]
fdnEqnMaskN = [ 1, -2]
fdMat = np.zeros([nNodes, nNodes])
fdMat[0, 0:2] = fdEqnMaskI
fdMat[1, 0:] = fdEqnMaskJ
fdMat[2, 1:] = fdnEqnMaskN
rhVec = -1 * np.ones(nNodes) * hElement**2
uAtX = solveBySpectralMethod(fdMat, rhVec)
uAtX_concat = list()
uAtX_concat.append(essentialBCAt0)
uAtX_concat += uAtX.tolist()
uAtX_concat.append(essentialBCAt1)
fig = plt.figure()
fdPlotHandle = plt.plot(exampleXCoord, uAtX_concat, label='fdPlot')
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
analyticPlotHandle = plt.plot(exampleXCoord, uAtXAnalytic, label='analyticPlot')
plt.legend(handles = [fdPlotHandle, analyticPlotHandle])
plt.show()
import sys; print('%s %s' % (sys.executable or sys.platform, sys.version))
import numpy as np
from matplotlib import pyplot as plt
import random
def solveBySpectralMethod(K, f):
    """Step 1
    w -> eigenvalues of K
    v -> eigenvectors of K"""
    w,v = np.linalg.eig(K)
    print "Dot product of column 0 and column 1: ", np.dot(v[:,0], v[:,1])
    print "Dot product of column 1 and column 2: ", np.dot(v[:,1], v[:,2])
    print "Dot product of column 0 and column 2: ", np.dot(v[:,0], v[:,2])
    print "Inner product of column 0 and column 1: ", innerproduct(v[:,0], v[:,1])
    print "Inner product of column 1 and column 2: ", innerproduct(v[:,1], v[:,2])
    print "Inner product of column 0 and column 2: ", innerproduct(v[:,0], v[:,2])
    print 'Checking orthogonality of eigenvectors...\n'
    c_i = 0.0
    for i in range(v.shape[1]):
        c_i += np.dot(v[:,i], w)/(np.dot(v[:,i], v[:,i]) * w)
    u = sum(c_i * v)
    return u
def innerproduct(vector1, vector2):
    return sum(vector1 * vector2)
nE1 = 4.0
domainX = [0.0, 1.0]
essentialBCAt0 = 0.0
essentialBCAt1 = 0.0
hElement = (domainX[1] - domainX[0])/nE1
i = 0
exampleXCoord = list()
while i <= 1:
    exampleXCoord.append(i)
    i += hElement
nNodes = nE1 - 1
fdEqnMaskI = [-2, 1]
fdEqnMaskJ = [1, -2, 1]
fdnEqnMaskN = [ 1, -2]
fdMat = np.zeros([nNodes, nNodes])
fdMat[0, 0:2] = fdEqnMaskI
fdMat[1, 0:] = fdEqnMaskJ
fdMat[2, 1:] = fdnEqnMaskN
rhVec = -1 * np.ones(nNodes) * hElement**2
uAtX = solveBySpectralMethod(fdMat, rhVec)
uAtX_concat = list()
uAtX_concat.append(essentialBCAt0)
uAtX_concat += uAtX.tolist()
uAtX_concat.append(essentialBCAt1)
fig = plt.figure()
fdPlotHandle = plt.plot(exampleXCoord, uAtX_concat, label='fdPlot')
uAtXAnalytic = .5 * (exampleXCoord - np.power(exampleXCoord,2))
analyticPlotHandle = plt.plot(exampleXCoord, uAtXAnalytic, label='analyticPlot')
plt.legend(handles = [fdPlotHandle, analyticPlotHandle])
plt.legend()#handles = [fdPlotHandle, analyticPlotHandle])
