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
from sklearn.cluster import KMeans
import time

repetitions = 250

test_score, train_score = [], []

start = time.time()

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

dgms = [np.load(path_file + u + '.npy') for u in files_list]

for _ in range(repetitions):
    
    train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms) 
    
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                         #("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                         ("TDA",       gd.representations.Entropy(mode = 'vector')),
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
                 "TDA__resolution":     [[20, 20], [40,40] ],
                 "TDA__bandwidth":      [0.005, 0.01, 0.1, 0.05],
                 "Estimator":           [SVC()]},
                
                #{"Scaler__use":         [True],
                #"TDA":                 [gd.representations.Landscape()], 
                #"TDA__resolution":     [100],
                #"Estimator":           [SVC]},
                
                #{"Scalar__use": [True],
                #"TDA" : [gd.representations.Atol(quantiser=KMeans(n_clusters=2, random_state=202006))]
                #"Estimator" : [SVC()]}
                
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