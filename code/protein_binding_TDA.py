import numpy as np
import pandas as pd
import gudhi as gd  
from sklearn import manifold
from pylab import *
import matplotlib.pyplot as plt
from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
import gudhi.representations
from ripser import Rips
from sklearn.model_selection import train_test_split
import time
# import sklearn_tda

path_file = "./datasets/"
files_list = [
    '1anf.corr_1.txt',  
    '1fqc.corr_2.txt',
    '1fqd.corr_3.txt',     
    '1mpd.corr_4.txt',    
    '3hpi.corr_5.txt', 
    '3mbp.corr_6.txt', 
    '4mbp.corr_7.txt',
    '1ez9.corr_1.txt', 
    '1fqa.corr_2.txt', 
    '1fqb.corr_3.txt',
    '1jw4.corr_4.txt',
    '1jw5.corr_5.txt', 
    '1lls.corr_6.txt',
    '1omp.corr_7.txt',
]

labs = [0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1
    ]

#0 stands for closed and 1 stands for open, the first 7 files are closed and the last 7 are open 

corr_list = [pd.read_csv(path_file + u,
                         header = None,
                         delim_whitespace = True) for u in files_list]

dist_list = [ 1- np.abs(c) for c in corr_list]

dgms = []

for i in range(14):
    D = dist_list[i]
    skeleton_protein = gd.RipsComplex(distance_matrix = D.values, max_edge_length = 0.8)
    Rips_simplex_tree_protein = skeleton_protein.create_simplex_tree(max_dimension = 3)
    BarCodes_Rips = Rips_simplex_tree_protein.persistence()
    dgms.append(Rips_simplex_tree_protein.persistence_intervals_in_dimension(1))
    

np.savetxt('PD_variable.txt', dgms)
    
train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms)  
    