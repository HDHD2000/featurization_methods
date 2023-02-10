from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
from gudhi.datasets.generators import points
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import time
from persistence_methods import carlsson_coordinates

start = time.time()

#the whole process is repeated 250 times

repetitions = 250

test_score, train_score = [], []

#=================================================#

#Collecting the persistence diagrams constructed using the code in 'protein_binding_tda'

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
#collecting all diagrams in one list

dgms = [np.load(path_file + u + '.npy') for u in files_list]

#=========================================================#

#Classifying the persistence diagrams using different featurization methods

for _ in range(repetitions):
    
    train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms) #randomly splits up the data into a training and testing set
    
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                         #("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                         ("TDA",       gd.representations.PersistenceFisherKernel(bandwidth = 0.01)), #change the featurization method here with the recommended parameters described down below
                         ("Estimator", SVC(C=5))]) #change the constant 'C' w.r.t. the recommended parameters down below
    
    model = pipe.fit(train_dgms, train_labs)
    train_score.append(model.score(train_dgms, train_labs))
    test_score.append(model.score(test_dgms,  test_labs))

print('Average training score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(train_score)))
print('Average testing score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(test_score)))


##==========================================================##
##CODE FOR CARLSSON COORDINATES

"""

for _ in range(repetitions):
    
    train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms) #randomly splits up the data into a training and testing set
    
    X_train_features_cc1, X_train_features_cc2, X_train_features_cc3, X_train_features_cc4, X_test_features_cc1, X_test_features_cc2, X_test_features_cc3, X_test_features_cc4 = carlsson_coordinates(train_dgms, test_dgms)
    
    X_train_features = np.column_stack((scale(X_train_features_cc1), scale(X_train_features_cc2), scale(X_train_features_cc3), scale(X_train_features_cc4)))
    X_test_features = np.column_stack((scale(X_test_features_cc1), scale(X_test_features_cc2), scale(X_test_features_cc3), scale(X_test_features_cc4)))
    clf = SVC(C=10).fit(X_train_features, train_labs)

    train_score.append(clf.score(X_train_features, train_labs))
    test_score.append(clf.score(X_test_features, test_labs))

print('Average training score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(train_score)))
print('Average testing score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(test_score)))
    
"""
#======================================================#

#Checking the time it took to compute everything

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")

#========================================================#
#Parameters used for the different featurization methods

"""

'BEST' PARAMETERS:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = default
       - bandwidth = default
       - SVC constant = default
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = default
       - SVC constant = default
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth = 0.1
       - SVC constant = 5
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = default
       - bandwidth = 0.01
       - SVC constant = 5
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 7
       - resolution = default
       - SVC constant = 5
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = [30,30]
       - bandwidth = 0.01
       - weight = lambda x: x[1]**2
       - SVC constant = 1
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = default
       - weight = default
       - SVC constant = 5
    
    Carlsson Coordinates: SVC = 10
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = default
       - mode = 'vector'
       - SVC constant = 10

"""