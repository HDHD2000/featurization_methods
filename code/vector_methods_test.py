# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 08:52:06 2022

@author: hdruenne
"""

from sklearn.cluster import KMeans
from gudhi.representations.vector_methods import Atol
import numpy as np
a = np.array([[1, 2, 4], [1, 4, 0], [1, 0, 4]])
b = np.array([[4, 2, 0], [4, 4, 0], [4, 0, 2]])
c = np.array([[3, 2, -1], [1, 2, -1]])
atol_vectoriser = Atol(quantiser=KMeans(n_clusters=2, random_state=202006))
atol_vectoriser.fit(X=[a, b, c]).centers 
# array([[ 2.        ,  0.66666667,  3.33333333],
#        [ 2.6       ,  2.8       , -0.4       ]])
atol_vectoriser(a)
# array([1.18168665, 0.42375966]) 
atol_vectoriser(c)
# array([0.02062512, 1.25157463]) 
atol_vectoriser.transform(X=[a, b, c]) 
# array([[1.18168665, 0.42375966],
#        [0.29861028, 1.06330156],
#        [0.02062512, 1.25157463]])