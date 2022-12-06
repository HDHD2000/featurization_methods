#import the filtration functions
from turkes_auxiliary_functions import binary_filtration_function, greyscale_filtration_function, density_filtration_function, radial_filtration_function, distance_filtration_function, dtm_filtration_function

#import the PD function calculator
from turkes_auxiliary_functions import pers_intervals_across_homdims

#import plotting functions
from turkes_auxiliary_functions import plot_image, plot_PD, plot_PI, plot_PL

#import the mnist data set
from keras.datasets import mnist

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

from scipy.spatial.distance import directed_hausdorff

import matplotlib.pyplot as plt # plt.matshow(), plt.scatter(), plt.hist(), etc.

import time # time.time()
import pickle # pickle.dump(), pickle.load()
from numba import njit, prange # @njit(parallel = True), prange() 
from joblib import Parallel, delayed # Parallel(n_jobs = -1)(delayed(function)(arguments) for arguments in arguments_array)

##-----------------------------------------------------##

repetitions = 1

test_score, train_score = [], []

start = time.time()

# Load data.
(train_data_, train_labels_), (test_data_, test_labels_) = mnist.load_data()
num_data_train = train_labels_.size
num_data_test = test_labels_.size

# For testing purposes, we can consider a smaller subset of the dataset.

for _ in range(repetitions):
    
    
    train_subset_size = 1000
    l_train = [random.randint(0,num_data_train) for i in range(train_subset_size)]
    train_data = train_data_[l_train,:]
    train_labels = train_labels_[l_train]
    
    test_subset_size = 200
    l_test = [random.randint(0,num_data_test) for i in range(test_subset_size)]
    test_data = test_data_[l_test, :]
    test_labels = test_labels_[l_test]
    
    # Calculate some auxiliary variables used throughout the notebook.
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
    
    
    ##----------------------------------------------------##
    
    # Choose filtration function.
    filt_temp = "Rips"
    
    # Choose filtration function parameters and calculate filtration function values for the training value
    if filt_temp == "binary":
        filt_func_vals_train = binary_filtration_function(train_data, 0.5)
        filt_func_vals_test = binary_filtration_function(test_data, 0.5)
    if filt_temp == "grsc":
        filt_func_vals_train = greyscale_filtration_function(train_data)
        filt_func_vals_test = greyscale_filtration_function(test_data)
    if filt_temp == "density":
        filt_func_vals_train = density_filtration_function(train_data, 0.5, 1) 
        filt_func_vals_test = density_filtration_function(test_data, 0.5, 1)
    if filt_temp == "radial":
        filt_func_vals_train= radial_filtration_function(train_data, 0.5, 0, 0)
        filt_func_vals_test= radial_filtration_function(test_data, 0.5, 0, 0)
    if filt_temp == "Rips":
        filt_func_vals_train = distance_filtration_function(train_data, 0.5)
        filt_func_vals_test = distance_filtration_function(test_data, 0.5)
    if filt_temp == "DTM":
        filt_func_vals_train = dtm_filtration_function(train_data, 0.5, 0.05)
        filt_func_vals_test = dtm_filtration_function(test_data, 0.5, 0.05)
    
    ##-----------------------------------------------------##
    
    # Choose PD parameters (for Rips and DTM) and calculate PDs.
    PDs0_train, PDs1_train = pers_intervals_across_homdims(filt_func_vals_train, filt_temp, train_data, 0.5)
    PDs0_test, PDs1_test = pers_intervals_across_homdims(filt_func_vals_test, filt_temp, test_data, 0.5)    
    
    # Choose homological dimension.
    train_dgms = PDs1_train
    test_dgms = PDs1_test
    
    # Calculate maximum value in PDs, to define PI bandwidth
    # and to determine the range of x and y-axis in PD plot.
    max_PDs = 0
    for PD in train_dgms:
        max_PD = np.max(PD)
        if max_PD > max_PDs:
            max_PDs = max_PD        
    
    ##---------------------------------------------------------##
    
    pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                             #("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                             ("TDA",       gd.representations.Landscape(num_landscapes=10)),
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