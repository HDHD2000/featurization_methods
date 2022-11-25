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

repetitions = 1000

num_pts = 20
num_diag_per_class = 20
ambient_dim_sphere = 3
dim_torus = 2

test_score, train_score = [], []

for _ in range(repetitions):
    dgms, labs = [], []
    for _ in range(num_diag_per_class):
        labs.append(0)
        sphere_points = points.sphere(n_samples = num_pts, ambient_dim = ambient_dim_sphere, radius = 1, sample = "random")
        AlphaSphere = gd.AlphaComplex(points = sphere_points).create_simplex_tree()
        dgmSphere = AlphaSphere.persistence()
        dgms.append(AlphaSphere.persistence_intervals_in_dimension(1))
        
    for _ in range(num_diag_per_class):
        labs.append(1)
        torus_points = points.torus(n_samples = num_pts, dim = dim_torus, sample = "random")
        AlphaTorus = gd.AlphaComplex(points = torus_points).create_simplex_tree()
        dgmTorus = AlphaTorus.persistence()
        dgms.append(AlphaTorus.persistence_intervals_in_dimension(1))
    
    train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms)    
    
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                     ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                     ("TDA",       gd.representations.Landscape()),
                     ("Estimator", SVC())])

    model = pipe.fit(train_dgms, train_labs)
    train_score.append(model.score(train_dgms, train_labs))
    test_score.append(model.score(test_dgms,  test_labs))

print('Average training score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(train_score)))
print('Average testing score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(test_score)))