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

num_pts = 50
num_diag_per_class = 50
ambient_dim_sphere = 3
dim_torus = 2

dgms, labs = [], []
for _ in range(num_diag_per_class):
    labs.append(0)
    sphere_points = points.sphere(n_samples = num_pts, ambient_dim = ambient_dim_sphere, radius = 1, sample = "random")
    RipsSphere = gd.RipsComplex(points = sphere_points).create_simplex_tree(max_dimension = 4)
    dgmSphere = RipsSphere.persistence()
    dgms.append(RipsSphere.persistence_intervals_in_dimension(1))
    
for _ in range(num_diag_per_class):
    labs.append(1)
    torus_points = points.torus(n_samples = num_pts, dim = dim_torus, sample = "random")
    RipsTorus = gd.RipsComplex(points = torus_points).create_simplex_tree(max_dimension = 4)
    dgmTorus = RipsTorus.persistence()
    dgms.append(RipsTorus.persistence_intervals_in_dimension(1))
    
test_size = 0.2
perm = np.random.permutation(len(labs))
limit = int(test_size * len(labs))
test_sub, train_sub = perm[:limit], perm[limit:]
train_labs = np.array(labs)[train_sub]
test_labs = np.array(labs)[test_sub]
train_dgms = [dgms[i] for i in train_sub]
test_dgms = [dgms[i] for i in test_sub]

pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                 ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                 ("TDA",       gd.representations.PersistenceImage()),
                 ("Estimator", SVC())])

# Parameters of pipeline. This is the place where you specify the methods you want to use to handle diagrams
param =    [{"Scaler__use":         [True],
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