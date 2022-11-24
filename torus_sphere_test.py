# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 08:58:15 2022

@author: hdruenne
"""

import numpy as np
from gudhi.datasets.generators import points
from ripser import Rips
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.representations

rips=Rips(maxdim=3)

# Generate 50 points on a sphere in R^4
sphere_points = points.sphere(n_samples = 50, ambient_dim = 3, radius = 1, sample = "random")

# Generate 50 points randomly on a torus in R^4
torus_points = points.torus(n_samples = 50, dim = 2, sample = "random")

#Rips Complexes
RipsSphere = gd.RipsComplex(points = sphere_points).create_simplex_tree(max_dimension = 4)
RipsTorus = gd.RipsComplex(points = torus_points).create_simplex_tree(max_dimension = 4)

#Persistence Diagrams
dgmSphere = RipsSphere.persistence()
dgmTorus = RipsTorus.persistence()

gd.plot_persistence_diagram(dgmSphere)
plt.show()

gd.plot_persistence_diagram(dgmTorus)
plt.show()


#Persistence Landscapes
PLSphere = gd.representations.Landscape(num_landscapes=5,resolution=100).fit_transform([RipsSphere.persistence_intervals_in_dimension(1)])
PLTorus = gd.representations.Landscape(num_landscapes=5,resolution=100).fit_transform([RipsTorus.persistence_intervals_in_dimension(1)])

plt.plot(PLSphere[0][:100])
plt.plot(PLSphere[0][100:200])
plt.plot(PLSphere[0][200:300])
plt.title("Landscape Sphere")
plt.show()

plt.plot(PLTorus[0][:100])
plt.plot(PLTorus[0][100:200])
plt.plot(PLTorus[0][200:300])
plt.title("Landscape Torus")
plt.show()

#Persistence Images
PISphere = gd.representations.PersistenceImage(0.005,weight=lambda x: x[1]**2, resolution=[20,20]).fit_transform([RipsSphere.persistence_intervals_in_dimension(1)])
PITorus = gd.representations.PersistenceImage(0.005,weight=lambda x: x[1]**2, resolution=[20,20]).fit_transform([RipsTorus.persistence_intervals_in_dimension(1)])

plt.imshow(np.flip(np.reshape(PISphere[0], [20,20]), 0))
plt.title("Persistence Image Sphere")
plt.show()

plt.imshow(np.flip(np.reshape(PITorus[0], [20,20]), 0))
plt.title("Persistence Image Torus")
plt.show()