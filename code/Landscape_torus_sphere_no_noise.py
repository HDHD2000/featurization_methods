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

repetitions = 20

nb_points = 1000
num_diag_per_class = 50

mu, sigma = 0, 0.5

test_score, train_score = [], []

start = time.time()

for _ in range(repetitions):
    dgms, labs = [], []
    for _ in range(num_diag_per_class):
        labs.append(0)
        sphere_points = points.sphere(n_samples = nb_points, ambient_dim = 3, radius = 1, sample = "random") 
        AlphaSphere = gd.AlphaComplex(points = sphere_points).create_simplex_tree()
        dgmSphere = AlphaSphere.persistence()
        dgms.append(AlphaSphere.persistence_intervals_in_dimension(1))
        
    for _ in range(num_diag_per_class):
        labs.append(1)
        alpha = 2*np.pi*np.random.random(nb_points)
        beta =  2*np.pi*np.random.random(nb_points) 
        torus_points = np.column_stack(((0.5*np.cos(alpha)+1)*np.cos(beta), (0.5*np.cos(alpha)+1)*np.sin(beta), 0.5*np.sin(alpha)))
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

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")