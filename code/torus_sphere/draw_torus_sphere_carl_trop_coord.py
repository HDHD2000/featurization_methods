import numpy as np
from gudhi.datasets.generators import points
from ripser import Rips
import matplotlib.pyplot as plt
import gudhi as gd
import pandas as pd
import gudhi.representations

def carlsson_coordinates(X):
    n = len(X)
    X_cc1 = np.zeros(n)
    X_cc2 = np.zeros(n)
    X_cc3 = np.zeros(n)
    X_cc4 = np.zeros(n)
    ymax = 0
    for i in range(0,n):
        if len(X[i])>0:
            b = np.max(X[i][:,1])
        else:
            b = ymax
        if ymax < b:
            ymax = b
        else:
            ymax = ymax
    for i in range(0,n):
        if len(X[i])>0:
            x = X[i][:,0]
            y = X[i][:,1]
            X_cc1[i] = sum(x*(y-x))
            X_cc2[i] = sum((ymax - y)*(y-x))
            X_cc3[i] = sum(x**2*(y-x)**4)
            X_cc4[i] = sum((ymax-y)**2*(y-x)**4)
        else:
            X_cc1[i] = 0
            X_cc2[i] = 0
            X_cc3[i] = 0
            X_cc4[i] = 0
    return np.array([X_cc1, X_cc2, X_cc3, X_cc4])

def tropical_coordinates(X):
    n = len(X)
    X_tc1 = np.zeros(n)
    X_tc3 = np.zeros(n)
    X_tc4 = np.zeros(n)
    X_tc5 = np.zeros(n)
    X_tc7 = np.zeros(n)
    for i in range(0,n):
        m = len(X[i])
        if m>0:
            sum_max2 = 0
            sub = np.zeros(m)
            x = X[i][:,0]
            y = X[i][:,1]
            X_tc1[i] = max(y-x)
            X_tc3[i] = sum(y-x)
            for j in range(0,m):
                sub[j] = min(28*(y[j] - x[j]), x[j])
            X_tc7[i] = sum(sub)
            max_sub = max(sub + y-x)
            X_tc4[i] = sum(max_sub - sub)
            for q in range(0,m):
                if q>0:
                    for k in range(0,q-1):
                        sum2 = y[q] - x[q] + y[k] - x[k]
                        if sum2>sum_max2:
                            sum_max2 = sum2
                    X_tc5[i] = sum_max2
                else:
                    X_tc5[i] = 0
        else:
            X_tc1[i] = 0
            X_tc3[i] = 0
            X_tc7[i] = 0
            X_tc4[i] = 0
            X_tc5[i] = 0
            
    return np.array([X_tc1,X_tc3,X_tc4,X_tc5,X_tc7])

#Fixing the Rips complex dimension
rips=Rips(maxdim=3)

#Parameter for the sample points
nb_samples = 1000
torus_dimension = 2
nb_objects = 50
sphere_dgms = []
torus_dgms = []

for _ in range(nb_objects):
    sphere_points = points.sphere(n_samples = nb_samples, ambient_dim = 3, radius = 1, sample = "random")

    alpha = 2*np.pi*np.random.random(nb_samples)
    beta =  2*np.pi*np.random.random(nb_samples) 
    torus_points = np.column_stack(((0.5*np.cos(alpha)+1)*np.cos(beta), (0.5*np.cos(alpha)+1)*np.sin(beta), 0.5*np.sin(alpha))) 
    
    #Rips Complexes
    AlphaSphere = gd.AlphaComplex(points = sphere_points).create_simplex_tree()
    AlphaTorus = gd.AlphaComplex(points = torus_points).create_simplex_tree()

    #Persistence Diagrams
    dgmSphere = AlphaSphere.persistence()
    dgmTorus = AlphaTorus.persistence()

    sphere_dgms.append(AlphaSphere.persistence_intervals_in_dimension(1))
    torus_dgms.append(AlphaTorus.persistence_intervals_in_dimension(1))

#Carlsson Coordinates (the three first coordinates so that they can be seen in 3D)
#ccSphere = carlsson_coordinates(sphere_dgms)
#ccTorus = carlsson_coordinates(torus_dgms)

tpSphere = tropical_coordinates(sphere_dgms)
tpTorus = tropical_coordinates(torus_dgms)

col1="XSphere"
col2="YSphere"
col3="ZSphere"
col4="XTorus"
col5="YTorus"
col6="ZTorus"

#data_cc = pd.DataFrame({col1:ccSphere[0], col2:ccSphere[1], col3:ccSphere[2], col4:ccTorus[0], col5:ccTorus[1], col6:ccTorus[2]})
#data_cc.to_excel('draw_torus_sphere.xlsx', sheet_name = 'sheet1', index = False)

data_tp = pd.DataFrame({col1:tpSphere[0], col2:tpSphere[1], col3:tpSphere[2], col4:tpTorus[0], col5:tpTorus[1], col6:tpTorus[2]})
data_tp.to_excel('draw_torus_sphere.xlsx', sheet_name = 'sheet2', index = False)