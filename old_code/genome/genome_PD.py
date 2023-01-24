import numpy as np
from ripser import Rips
import pandas as pd

rips = Rips(maxdim=2)

genome = pd.read_csv('genome_data.txt', header = None, delim_whitespace= True)
l_ = int(np.max(genome)[0])
genome = genome.to_numpy()

D_matrix = np.zeros((l_, l_))

for i in range(len(genome)):
    D_matrix[int(genome[i][0])-1][int(genome[i][1])-1] = genome[i][2]
    
diagrams = rips.fit_transform(D_matrix, distance_matrix = True)
rips.plot(diagrams)