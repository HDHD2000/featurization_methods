# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:01:21 2022

@author: hdruenne
"""

import numpy as np
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
from ripser import Rips

rips=Rips()

num_pts = 1000
r       = 2.5

X = np.empty([num_pts,2])
x, y = np.random.uniform(), np.random.uniform()
for i in range(num_pts):
    X[i,:] = [x, y]
    x = (X[i,0] + r * X[i,1] * (1-X[i,1])) % 1.
    y = (X[i,1] + r * x * (1-x)) % 1.
    
plt.scatter(X[:,0], X[:,1], s=3)
plt.show()

acX = gd.AlphaComplex(points=X).create_simplex_tree()
dgmX = acX.persistence()

gd.plot_persistence_diagram(dgmX)
plt.show()

LS = gd.representations.Landscape(resolution=100)
L = LS.fit_transform([acX.persistence_intervals_in_dimension(1)])

plt.plot(L[0][:100])
plt.plot(L[0][100:200])
plt.plot(L[0][200:300])
plt.title("Landscape")
plt.show()

PI = gd.representations.PersistenceImage(bandwidth=1e-4, weight=lambda x: x[1]**2, \
                                         im_range=[0,.004,0,.004], resolution=[20,20])
pi = PI.fit_transform([acX.persistence_intervals_in_dimension(1)])

plt.imshow(np.flip(np.reshape(pi[0], [20,20]), 0))
plt.title("Persistence Image")
plt.show()

h = 3.5
Y = np.empty([num_pts,2])
x, y = np.random.uniform(), np.random.uniform()
for i in range(num_pts):
    Y[i,:] = [x, y]
    x = (Y[i,0] + h * Y[i,1] * (1-Y[i,1])) % 1.
    y = (Y[i,1] + h * x * (1-x)) % 1.
    
plt.scatter(Y[:,0], Y[:,1], s=3)
plt.show()
    
acY = gd.AlphaComplex(points=Y).create_simplex_tree()
dgmY = acY.persistence()

gd.plot_persistence_diagram(dgmY)
plt.show()

LSY = gd.representations.Landscape(resolution=100)
LY = LSY.fit_transform([acY.persistence_intervals_in_dimension(1)])

plt.plot(LY[0][:100])
plt.plot(LY[0][100:200])
plt.plot(LY[0][200:300])
plt.title("Landscape Y")
plt.show()

PI = gd.representations.PersistenceImage(bandwidth=1e-4, weight=lambda x: x[1]**2, \
                                         im_range=[0,.004,0,.004], resolution=[20,20])
pi = PI.fit_transform([acY.persistence_intervals_in_dimension(1)])

plt.imshow(np.flip(np.reshape(pi[0], [20,20]), 0))
plt.title("Persistence Image")
plt.show()

#PWG = gd.representations.PersistenceWeightedGaussianKernel(bandwidth=0.01, kernel_approx=None,\
                                        #weight=lambda x: np.arctan(np.power(x[1], 1)))
#PWG.fit([acX.persistence_intervals_in_dimension(1)])
#pwg = PWG.transform([acY.persistence_intervals_in_dimension(1)])
#print("PWG kernel is " + str(pwg[0][0]))

#PSS = gd.representations.PersistenceScaleSpaceKernel(bandwidth=1.)
#PSS.fit([acX.persistence_intervals_in_dimension(1)])
#pss = PSS.transform([acY.persistence_intervals_in_dimension(1)])
#print("PSS kernel is " + str(pss[0][0]))

#PF = gd.representations.PersistenceFisherKernel(bandwidth_fisher=.001, bandwidth=.001, kernel_approx=None)
#PF.fit([acX.persistence_intervals_in_dimension(1)])
#pf = PF.transform([acY.persistence_intervals_in_dimension(1)])
#print("PF kernel is " + str(pf[0][0]))

#SW = gd.representations.SlicedWassersteinKernel(bandwidth=.1, num_directions=100)
#SW.fit([acX.persistence_intervals_in_dimension(1)])
#sw = SW.transform([acY.persistence_intervals_in_dimension(1)])
#print("SW kernel is " + str(sw[0][0]))

#BD = gd.representations.BottleneckDistance(epsilon=.001)
#BD.fit([acX.persistence_intervals_in_dimension(1)])
#bd = BD.transform([acY.persistence_intervals_in_dimension(1)])
#print("Bottleneck distance is " + str(bd[0][0]))

#WD = gd.representations.WassersteinDistance(internal_p=2, order=2)
#WD.fit([acX.persistence_intervals_in_dimension(1)])
#wd = WD.transform([acY.persistence_intervals_in_dimension(1)])
#print("Wasserstein distance is " + str(wd[0][0]))