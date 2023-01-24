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

nb_samples = 1000
torus_dimension = 2

#Gaussian Noise
mu, sigma = 0, 0.5

# Generate 50 points on a sphere in R^3
sphere_points = points.sphere(n_samples = nb_samples, ambient_dim = 3, radius = 1, sample = "random")

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
s = list(zip(*sphere_points))
ax.scatter3D(s[0], s[1], s[2])
plt.show()

# Generate 50 points randomly on a torus
#torus_points = points.torus(n_samples = nb_samples, dim = 2, sample = "random")

alpha = 2*np.pi*np.random.random(nb_samples)
beta =  2*np.pi*np.random.random(nb_samples) 
torus_points = np.column_stack(((0.5*np.cos(alpha)+1)*np.cos(beta), (0.5*np.cos(alpha)+1)*np.sin(beta), 0.5*np.sin(alpha))) 

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
t = list(zip(*torus_points))
ax.scatter3D(t[0],t[1],t[2])
plt.show()

#Rips Complexes
AlphaSphere = gd.AlphaComplex(points = sphere_points).create_simplex_tree()
AlphaTorus = gd.AlphaComplex(points = torus_points).create_simplex_tree()

#Persistence Diagrams
dgmSphere = AlphaSphere.persistence()
dgmTorus = AlphaTorus.persistence()

gd.plot_persistence_diagram(dgmSphere)
plt.show()

gd.plot_persistence_diagram(dgmTorus)
plt.show()


#Persistence Landscapes
PLSphere = gd.representations.Landscape(num_landscapes=5,resolution=100).fit_transform([AlphaSphere.persistence_intervals_in_dimension(1)])
PLTorus = gd.representations.Landscape(num_landscapes=5,resolution=100).fit_transform([AlphaTorus.persistence_intervals_in_dimension(1)])

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
PISphere = gd.representations.PersistenceImage(0.005,weight=lambda x: x[1]**2, resolution=[20,20]).fit_transform([AlphaSphere.persistence_intervals_in_dimension(1)])
PITorus = gd.representations.PersistenceImage(0.005,weight=lambda x: x[1]**2, resolution=[20,20]).fit_transform([AlphaTorus.persistence_intervals_in_dimension(1)])

plt.imshow(np.flip(np.reshape(PISphere[0], [20,20]), 0))
plt.title("Persistence Image Sphere")
plt.show()

plt.imshow(np.flip(np.reshape(PITorus[0], [20,20]), 0))
plt.title("Persistence Image Torus")
plt.show()

#Persistent Silhouette
PSSphere = gd.representations.Silhouette(resolution = 100).fit_transform([AlphaSphere.persistence_intervals_in_dimension(1)])
PSTorus = gd.representations.Silhouette(resolution = 100).fit_transform([AlphaTorus.persistence_intervals_in_dimension(1)])

print(PSSphere)


plt.plot(PSSphere[0][:100])
plt.title("Silhouette Sphere")
plt.show()

plt.plot(PSTorus[0][:100])
plt.title("Silhouette Torus")
plt.show()

#Persistent Entropy
PESphere = gd.representations.Entropy(mode="vector", resolution = 100).fit_transform([AlphaSphere.persistence_intervals_in_dimension(1)])
PETorus = gd.representations.Entropy(mode="vector", resolution = 100).fit_transform([AlphaTorus.persistence_intervals_in_dimension(1)])

print(PESphere)

plt.plot(PESphere[0][:100])
plt.title("Entropy Sphere")
plt.show()

plt.plot(PETorus[0][:100])
plt.title("Entropy Torus")
plt.show()