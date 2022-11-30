from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
from ripser import Rips
from gudhi.datasets.generators import points
from sklearn.model_selection import train_test_split
import time

#Creating the data set: a family of diagrams coming from a discrete one-parameter dynamical system with varying parameter

num_pts = 1000
num_diag_per_class = 50

repetitions = 1

test_score, train_score = [], []

start = time.time()

for _ in range(repetitions):
    dgms, labs = [], []
    for idx, radius in enumerate([2.5, 3.5, 4., 4.1, 4.3]):
        for _ in range(num_diag_per_class):
            labs.append(idx)
            X = np.empty([num_pts,2])
            x, y = np.random.uniform(), np.random.uniform()
            for i in range(num_pts):
                X[i,:] = [x, y]
                x = (X[i,0] + radius * X[i,1] * (1-X[i,1])) % 1.
                y = (X[i,1] + radius * x * (1-x)) % 1.
            ac = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=1e12)
            dgm = ac.persistence()
            dgms.append(ac.persistence_intervals_in_dimension(1))
        
    train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms)  
    
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                     ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                     ("TDA",       gd.representations.PersistenceImage(bandwidth = 0.005, weight = lambda x: x[1]**2, resolution = [20,20])),
                     ("Estimator", SVC())])

    param =    [#{"Scaler__use":         [False],
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
            
            {"Scaler__use":         [True],
             "TDA":                 [gd.representations.PersistenceImage()], 
             "TDA__resolution":     [[5, 5], [6,6] ],
             "TDA__bandwidth":      [0.01, 0.1, 0.5, 1, 10],
             "Estimator":           [SVC()]},
            
            #{"Scaler__use":         [True],
            # "TDA":                 [gd.representations.Landscape()], 
            # "TDA__resolution":     [100],
            # "Estimator":           [SVC()]},
           
           ]

    model = pipe.fit(train_dgms, train_labs)
    train_score.append(model.score(train_dgms, train_labs))
    test_score.append(model.score(test_dgms,  test_labs))

print('Average training score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(train_score)))
print('Average testing score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(test_score)))

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")
