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


data_series, labels = load_UCR_UEA_dataset(name="WordSynonyms")

num_train = 267
num_test = 638

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
                             ("TDA",       gd.representations.PersistenceFisherKernel()), #change the featurization methods following the recommended values below
                             ("Estimator", SVC(kernel="precomputed", gamma="auto"))]) #change the constant 'C' following the recommendations below
                                # for kernel methods further add 'kernel="precomputed", gamma="auto"' in SVC()
model = pipe.fit(train_dgms, train_labels)
print('Training score: ' + str(model.score(train_dgms, train_labels)))
print('Testing score: ' + str(model.score(test_dgms,  test_labels)))

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")

##-----------------------------------------------##

#Parameters for the different time series data sets and for each featurization method

"""

'BEST' PARAMETERS:

ADIAC:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 

CHLORINE CONCENTRATION:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
COMPUTERS:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
ECGFIVEDAYS:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
LIGHTNING7:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
SHAPELETSIM:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
TRACE:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
CHINATOWN:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
GUNPOINTAGESPAN:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
PIGCVP:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 
    
WORDSYNONYMS:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth =
       - SVC constant =
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant =
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 
       - resolution = 
       - SVC constant = 
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = 
       - weight = 
       - SVC constant =
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 
       - mode = 'vector'
       - SVC constant = 

"""