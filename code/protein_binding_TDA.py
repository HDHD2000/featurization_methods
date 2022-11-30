import numpy as np
import pandas as pd
import gudhi as gd  
import gudhi.representations

path_file = "./datasets/"
files_list = [
    '1anf.corr_1',  
    '1fqc.corr_2',
    '1fqd.corr_3',     
    '1mpd.corr_4',    
    '3hpi.corr_5', 
    '3mbp.corr_6', 
    '4mbp.corr_7',
    '1ez9.corr_1', 
    '1fqa.corr_2', 
    '1fqb.corr_3',
    '1jw4.corr_4',
    '1jw5.corr_5', 
    '1lls.corr_6',
    '1omp.corr_7',
]


corr_list = [pd.read_csv(path_file + u + '.txt',
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
    np.save(path_file + files_list[i] + '.npy', dgms[i])
    