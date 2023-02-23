from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import gudhi as gd
import gudhi.representations
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
from persistence_methods import carlsson_coordinates, tropical_coordinates, top_sign

##--------------------------------------##
#Creating the data set: a family of diagrams coming from a discrete one-parameter dynamical system with varying parameter

num_pts = 250              #number of iterations of the dynamical system
num_diag_per_class = 25     #number of persistence diagrams per parameter value

repetitions = 10      #number of times the whole process is repeated

test_score, train_score = [], []

start = time.time()

##==============================================##
##Construction of the data set, computation of the persistence diagrams,
##  application of the featurization mehod and classification using SVC

"""

for _ in range(repetitions):
    dgms, labs = [], []
    for idx, radius in enumerate([2.5, 3.5, 4., 4.1, 4.3]): #the different parameter values are 2.5, 3.5, 4., 4.1 and 4.3
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
            dgms.append(ac.persistence_intervals_in_dimension(1)) ##list of all persistence diagrams
        
    train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms)   #dividing the set of persistence diagrams into a testing and a training set
    
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                     ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                     ("TDA",       gd.representations.Entropy(mode = 'vector', normalized = False, resolution = 200)), #change the featurization method with the recommended values below
                     ("Estimator", SVC(C=10))]) #change the constant 'C' following the recommendations below
                        #for kernel methods further add 'kernel="precomputed", gamma="auto"' in SVC()

    model = pipe.fit(train_dgms, train_labs)
    train_score.append(model.score(train_dgms, train_labs))
    test_score.append(model.score(test_dgms,  test_labs))

print('Average training score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(train_score)))
print('Average testing score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(test_score)))

"""

##===============================================================##

##CODE FOR CARLSSON AND TROPICAL COORDINATES



coordinate = 'signature' ##'tropical', 'signature' or 'carlsson' depending on the coordinates one wants to use

for _ in range(repetitions):
    dgms, labs = [], []
    for idx, radius in enumerate([2.5, 3.5, 4., 4.1, 4.3]): #the different parameter values are 2.5, 3.5, 4., 4.1 and 4.3
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
            dgms.append(ac.persistence_intervals_in_dimension(1)) ##list of all persistence diagrams
        
    train_labs, test_labs, train_dgms, test_dgms = train_test_split(labs, dgms) #randomly splits up the data into a training and testing set
    
    if coordinate == 'carlsson':
        X_train_features = carlsson_coordinates(train_dgms)
        X_test_features = carlsson_coordinates(test_dgms)
    
    if coordinate == 'tropical':
        X_train_features = tropical_coordinates(train_dgms)
        X_test_features = tropical_coordinates(test_dgms)  
        
    if coordinate == 'signature':
        X_train_features = top_sign(train_dgms,dim_trunc=3)
        X_test_features = top_sign(test_dgms,dim_trunc=3)
        
    clf = SVC(C=20).fit(X_train_features, train_labs)

    train_score.append(clf.score(X_train_features, train_labs))
    test_score.append(clf.score(X_test_features, test_labs))

print('Average training score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(train_score)))
print('Average testing score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(test_score)))

   
##================================================================##

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")

##----------------------------------------------##

#Parameters used for the different featurization methods

"""

'BEST' PARAMETERS:
    SWK: gd.representations.SlicedWassersteinKernel()
       - num_directions = default
       - bandwidth = 0.02
       - SVC constant = 10
    
    PWGK: gd.representations.PersistenceWeightedGaussianKernel()
       - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 0.01
       - SVC constant = 10
    
    PSSK: gd.representations.PersistenceScaleSpaceKernel()
       - bandwidth = 0.01
       - SVC constant = 10
    
    PFK : gd.representations.PersistenceFisherKernel()
       - bandwidth_fisher = default
       - bandwidth = 0.01
       - SVC constant = 10
       
    Landscape: gd.representations.Landscape()
       - num_landscapes = 10
       - resolution = 200
       - SVC constant = 10
       
    Persistence Images: gd.representations.PersistenceImage()
       - resolution = [40,40]
       - bandwidth = 0.001
       - weight = lambda x: x[1]**2
       - SVC constant = 10
       
    Persistence Silhouette: gd.representations.Silhouette()
       - resolution = default
       - weight = default
       - SVC constant = 10
    
    Carlsson Coordinates: SVC = 40
    
    Tropical Coordinates: SVC = 20
    
    Topological Signature: 
       - SVC = 20
       - truncated dimension = 3
    
    Persistent Entropy: gd.representations.Entropy()
       - resolution = 200
       - mode = 'vector'
       - normalized = False
       - SVC constant = 10

"""
