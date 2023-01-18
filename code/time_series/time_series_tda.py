from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sktime.datasets import load_UCR_UEA_dataset
import numpy as np
import gudhi as gd
import gudhi.representations
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
import pandas as pd

start = time.time()

##-------------------------------------------------------##
#EXTRACTING THE DATA SET


data_series, labels = load_UCR_UEA_dataset(name="Lightning7")

num_train = 70
num_test = 73

data_series = data_series.to_numpy()
data_series = data_series.flatten()

num_data_series = np.shape(data_series)[0]

##-------------------------------------------------------##

dgms = []

##Computing the persistence diagrams by first creating the 
##point clouds using the Takens embedding

for j in range(num_data_series):
    data_ = data_series[j].to_numpy()
    l_ = len(data_)
    data_points = np.zeros([l_ - 2,3])
    for i in range(l_-2):
        data_points[i, :] = [data_[i], data_[i+1], data_[i+2]]
    ac = gd.AlphaComplex(data_points).create_simplex_tree()
    dgm = ac.persistence()
    dgms.append(ac.persistence_intervals_in_dimension(1))

##-------------------------------------------------------##

#Embedding the persistence diagrams in a Hilbert space 
# and checking their classification rates

train_dgms = dgms[0:num_train-1]
train_labels = labels[0:num_train-1]
test_dgms = dgms[num_train:]
test_labels = labels[num_train:]

if num_train+num_test == num_data_series:
    print('ALL GOOD IN THE HOOD')

pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                             ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                             ("TDA",       gd.representations.PersistenceFisherKernel(bandwidth = 1)),
                             ("Estimator", SVC(kernel="precomputed", gamma="auto"))])
        
#param =    [#{"Scaler__use":         [False],
                    #"TDA":                 [gd.representations.SlicedWassersteinKernel()], 
                    #"TDA__bandwidth":      [0.1, 1.0],
                    # "TDA__num_directions": [20],
                    # "Estimator":           [SVC(kernel="precomputed", gamma="auto")]},
                    
                    #{"Scaler__use":         [False],
                    # "TDA":                 [gd.representations.PersistenceWeightedGaussianKernel()], 
                    #  "TDA__bandwidth":      [0.1, 0.01],
                    # "TDA__weight":         [lambda x: np.arctan(x[1]-x[0])], 
                    #"Estimator":           [SVC(kernel="precomputed", gamma="auto")]},
                    
                    #{"Scalar__use": [False],
                    #"TDA": [gd.representations.PersistenceScaleSpaceKernel()],
                    #"Estimator": [SVC(kernel = "precomputed", gamma="auto")]},
                    
                    #{"Scalar__use" : [False],
                    # "TDA": [gd.representations.PersistenceFisherKernel()],
                    #"Estimator": [SVC(kernel = "precomputed", gamma="auto")]},
                    
                    #{"Scaler__use":         [True],
                     #"TDA":                 [gd.representations.PersistenceImage()], 
                    # "TDA__resolution":     [[20, 20] ],
                     #"TDA__bandwidth":      [0.05 * max_PDs],
                     #"TDA_weight": [weight = lambda x: x[1]**2],
                     #"Estimator":           [SVC()]},
                    
                    #{"Scaler__use":         [True],
                    #"TDA":                 [gd.representations.Landscape()], 
                    #"TDA__resolution":     [100],
                    #"Estimator":           [SVC]},
                    
                    #{"Scalar__use": [True],
                    #"TDA" : [gd.representations.Atol(quantiser=KMeans(n_clusters=2, random_state=202006))]
                    #"Estimator" : [SVC()]} 
#      ]

model = pipe.fit(train_dgms, train_labels)
print('Training score: ' + str(model.score(train_dgms, train_labels)))
print('Testing score: ' + str(model.score(test_dgms,  test_labels)))

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")