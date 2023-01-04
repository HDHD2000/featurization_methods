#import the filtration functions
from sweeping import sweep_right_to_left_filtration, sweep_left_to_right_filtration, sweep_up_down_filtration, sweep_down_up_filtration, build_point_cloud

#import the PD function calculator
from sweeping import pers_intervals_across_homdims

#import plotting functions
from turkes_auxiliary_functions import plot_image, plot_PD, plot_PI, plot_PL

#import the mnist data set
from keras.datasets import mnist
from sklearn.ensemble        import RandomForestClassifier

#import additional libraries
import numpy as np # np.loadtxt(), np.sum(), np.max(), np.random.uniform(), etc.
import math # math.floor()
import random

from keras.preprocessing.image import ImageDataGenerator # ImageDataGenerator()

import gudhi as gd # gd.CubicalComplex(), gd.RipsComplex(), gd.SimplexTree(), gd.persistence(), gd.persistence_intervals_in_dimension(), gd.bottleneck_distance()
from gudhi.weighted_rips_complex import WeightedRipsComplex
import gudhi.wasserstein
import gudhi.representations # gd.representations.Landscape(), gd.representations.PersistenceImage()
# other TDA libraries:
# import sklearn_tda # Mathieu Carriere, https://github.com/MathieuCarriere/sklearn-tda 
# import gtda # Guillaume Tauzin, Umberto Lupo, Kathryn Hess, https://github.com/giotto-ai/giotto-tda
# import teaspoon # Elizabeth Munch, https://github.com/lizliz/teaspoon
# import scikit-tda # Chris Trailie, https://github.com/scikit-tda
# import homcloud.interface as hc # Ippei Obayashi, http://www.wpi-aimr.tohoku.ac.jp/hiraoka_labo/homcloud/index.en.html

from sklearn import datasets # datasets.load_digits()
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances 
from sklearn.metrics import plot_confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin # class kernel_rbf(BaseEstimator, TransformerMixin)
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone 
from tempfile import TemporaryDirectory
from sklearn.cluster import KMeans
from sklearn.preprocessing   import MinMaxScaler

from scipy.spatial.distance import directed_hausdorff

import matplotlib.pyplot as plt # plt.matshow(), plt.scatter(), plt.hist(), etc.

import time # time.time()
import pickle # pickle.dump(), pickle.load()
from numba import njit, prange # @njit(parallel = True), prange() 
from joblib import Parallel, delayed # Parallel(n_jobs = -1)(delayed(function)(arguments) for arguments in arguments_array)

##=================================================##

repetitions = 1

test_score, train_score = [], []

start = time.time()

# Load data.
(train_data_, train_labels_), (test_data_, test_labels_) = mnist.load_data()
num_data_train = train_labels_.size
num_data_test = test_labels_.size

for _ in range(repetitions):
    
    train_data = train_data_[0:999, :]
    test_data = test_data_[0:199,:]
    train_labels = train_labels_[0:999]
    test_labels = test_labels_[0:199]
    
    min_train_data = np.min(train_data)
    max_train_data = np.max(train_data)
    num_train_data_points = len(train_labels)
    num_test_data_points = len(test_labels)
    num_x_pixels = np.abs(train_data.shape[1]).astype(int)
    num_y_pixels = num_x_pixels
    num_pixels = num_x_pixels * num_y_pixels
    train_data = train_data.reshape((num_train_data_points, num_pixels))
    test_data = test_data.reshape((num_test_data_points, num_pixels))
    train_data_images = train_data.reshape((num_train_data_points, num_x_pixels, num_y_pixels))
    
    filt_func_vals_train_right_left = sweep_right_to_left_filtration(train_data)
    filt_func_vals_test_right_left = sweep_right_to_left_filtration(test_data)
    PDs1_train_right_left = pers_intervals_across_homdims(filt_func_vals_train_right_left, train_data, 0.5)
    PDs1_test_right_left = pers_intervals_across_homdims(filt_func_vals_test_right_left, test_data, 0.5) 
    
    filt_func_vals_train_left_right = sweep_left_to_right_filtration(train_data)
    filt_func_vals_test_left_right = sweep_left_to_right_filtration(test_data)
    PDs1_train_left_right = pers_intervals_across_homdims(filt_func_vals_train_left_right, train_data, 0.5)
    PDs1_test_left_right = pers_intervals_across_homdims(filt_func_vals_test_left_right, test_data, 0.5)    
    
    filt_func_vals_train_up_down = sweep_up_down_filtration(train_data)
    filt_func_vals_test_up_down = sweep_up_down_filtration(test_data)
    PDs1_train_up_down = pers_intervals_across_homdims(filt_func_vals_train_up_down, train_data, 0.5)
    PDs1_test_up_down = pers_intervals_across_homdims(filt_func_vals_test_up_down, test_data, 0.5)
    
    filt_func_vals_train_down_up = sweep_down_up_filtration(train_data)
    filt_func_vals_test_down_up = sweep_down_up_filtration(test_data)
    PDs1_train_down_up = pers_intervals_across_homdims(filt_func_vals_train_down_up, train_data, 0.5)
    PDs1_test_down_up = pers_intervals_across_homdims(filt_func_vals_test_down_up, test_data, 0.5)
    
    # Choose homological dimension.
    train_dgms = []
    test_dgms = []
    
    for i in range(num_train_data_points):
        train_dgms.append(np.concatenate((PDs1_train_left_right[i], PDs1_train_right_left[i], PDs1_train_up_down[i], PDs1_train_down_up[i])))
    for j in range(num_test_data_points):
        test_dgms.append(np.concatenate((PDs1_test_left_right[j], PDs1_test_right_left[j], PDs1_test_up_down[j], PDs1_test_down_up[j])))
    

    ##---------------------------------------------------------##
    
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                             ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                             ("TDA",       gd.representations.PersistenceWeightedGaussianKernel(bandwidth = 0.1, weight = lambda x: np.arctan(x[1]-x[0]))),
                             ("Estimator", SVC(kernel = "precomputed", gamma="auto"))])
        
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
                    
             ]
        
    model = pipe.fit(train_dgms, train_labels)
    train_score.append(model.score(train_dgms, train_labels))
    test_score.append(model.score(test_dgms,  test_labels))
    
print('Average training score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(train_score)))
print('Average testing score after ' + str(repetitions) + ' repetitions: ' + str(np.mean(test_score)))

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")