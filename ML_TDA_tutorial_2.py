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

rips=Rips()

#Creating the data set: a family of diagrams coming from a discrete one-parameter dynamical system with varying parameter

num_pts = 100
num_diag_per_class = 50

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
        
#Separating the training set from the test set
        
test_size            = 0.2
perm                 = np.random.permutation(len(labs))
limit                = int(test_size * len(labs))
test_sub, train_sub  = perm[:limit], perm[limit:]
train_labs           = np.array(labs)[train_sub]
test_labs            = np.array(labs)[test_sub]
train_dgms           = [dgms[i] for i in train_sub]
test_dgms            = [dgms[i] for i in test_sub]

# Definition of pipeline: which sequentially applies a list of transforms and an estimator
# DiagramSelector: selects all the values of the persistence diagram which are not essential
# DiagramScaler: scales the diagram into [0,1]
pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                 ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                 ("TDA",       gd.representations.PersistenceImage()),
                 ("Estimator", SVC())])

# Parameters of pipeline. This is the place where you specify the methods you want to use to handle diagrams
param =    [{"Scaler__use":         [False],
             "TDA":                 [gd.representations.SlicedWassersteinKernel()], 
             "TDA__bandwidth":      [0.1, 1.0],
             "TDA__num_directions": [20],
             "Estimator":           [SVC(kernel="precomputed", gamma="auto")]},
            
            {"Scaler__use":         [False],
             "TDA":                 [gd.representations.PersistenceWeightedGaussianKernel()], 
             "TDA__bandwidth":      [0.1, 0.01],
             "TDA__weight":         [lambda x: np.arctan(x[1]-x[0])], 
             "Estimator":           [SVC(kernel="precomputed", gamma="auto")]},
            
            {"Scaler__use":         [True],
             "TDA":                 [gd.representations.PersistenceImage()], 
             "TDA__resolution":     [[20, 20] ],
             "TDA__bandwidth":      [0.005, 0.01, 0.1],
             "Estimator":           [SVC()]},
            
            {"Scaler__use":         [True],
             "TDA":                 [gd.representations.Landscape()], 
             "TDA__resolution":     [100],
             "Estimator":           [RandomForestClassifier()]},
           
           ]

# Crossvalidation method is used to find the best parameters and the best model
model = GridSearchCV(pipe, param, cv=3)
model = model.fit(train_dgms, train_labs)
print(model.best_params_)
print("Train accuracy = " + str(model.score(train_dgms, train_labs)))
print("Test accuracy  = " + str(model.score(test_dgms,  test_labs)))
