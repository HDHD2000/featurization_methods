import gudhi as gd
import pandas as pd
import gudhi.representations
import matplotlib.pyplot as plt
from gudhi.datasets.generators import points
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np

path_file = "./datasets/"

data = pd.read_csv(path_file + '1fqc.corr_2.txt',
                         header = None,
                         delim_whitespace = True).to_numpy()

dist = 1 - np.abs(data)

dgms = ripser(dist, distance_matrix = True)['dgms']
plot_diagrams(dgms, show= True)