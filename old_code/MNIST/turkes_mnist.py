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

##--------------------------------------------------##

# Load data.
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# For testing purposes, we can consider a smaller subset of the dataset.

train_subset_size = 1000
train_data = train_data[0:train_subset_size, :]
train_labels = train_labels[0:train_subset_size]
test_subset_size = 200
test_data = test_data[0:test_subset_size, :]
test_labels = test_labels[0:test_subset_size]

# Calculate some auxiliary variables used throughout the notebook.
min_data = np.min(train_data)
max_data = np.max(train_data)
num_data_points = train_labels.size
num_x_pixels = np.abs(train_data.shape[1]).astype(int)
num_y_pixels = num_x_pixels
num_pixels = num_x_pixels * num_y_pixels
train_data = train_data.reshape((num_data_points, num_pixels))
train_data_images = train_data.reshape((num_data_points, num_x_pixels, num_y_pixels))

# Auxiliary cell to explore the dataset and visualize a few example images.

# Explore the dataset.
print("type(data) =", type(train_data))
print("data.shape =", train_data.shape, "\n")
print("type(labels) =", type(train_labels))
print("labels.shape =", train_labels.shape, "\n")
print("min_data =", min_data)
print("max_data =", max_data, "\n")
print("num_data_points =", num_data_points)
print("num_x_pixels =", num_x_pixels)
print("num_y_pixels =", num_y_pixels)
print("num_pixels =", num_pixels, "\n")
print("type(data_images) =", type(train_data_images))
print("data_images.shape =", train_data_images.shape, "\n")
# print("data_images[example_1] = \n", data_images[example_1])

##----------------------------------------------##

# Choose images.
data_1 = np.copy(train_data[0])
data_2 = np.copy(train_data[46])
data_temp = np.asarray([data_1,data_2])

# Choose filtration function.
filt_temp = "Rips"

# Choose filtration function parameters and calculate filtration function values.
if filt_temp == "binary":
    filt_func_vals_temp = binary_filtration_function(data_temp, 0.5)
if filt_temp == "grsc":
    filt_func_vals_temp = greyscale_filtration_function(data_temp)
if filt_temp == "density":
    filt_func_vals_temp = density_filtration_function(data_temp, 0.5, 1) 
if filt_temp == "radial":
    filt_func_vals_temp = radial_filtration_function(data_temp, 0.5, 0, 0)
if filt_temp == "Rips":
    filt_func_vals_temp = distance_filtration_function(data_temp, 0.5)
if filt_temp == "DTM":
    filt_func_vals_temp = dtm_filtration_function(data_temp, 0.5, 0.05)

# Choose PD parameters (for Rips and DTM) and calculate PDs.
PDs0, PDs1 = pers_intervals_across_homdims(filt_func_vals_temp, filt_temp, data_temp, 0.5)
    
# Choose homological dimension.
PDs_temp = PDs1
    
# Print persistence intervals for Table 1 in turkevs2021noise.
print("1-dim PH wrt", filt_temp, "filtration: \n \n")
for i, PD in enumerate(PDs_temp): 
    print("data point %d" %(i+1), ": \n", PD, "\n\n")
    

# Calculate maximum value in PDs, to define PI bandwidth
# and to determine the range of x and y-axis in PD plot.
max_PDs_temp = 0
for PD in PDs_temp:
    max_PD = np.max(PD)
    if max_PD > max_PDs_temp:
        max_PDs_temp = max_PD        
    
# Choose PL parameters and calculate PLs.
landscape_ = gd.representations.Landscape(num_landscapes = 10, resolution = 100)
PLs_temp = landscape_.fit_transform(PDs_temp)

# Choose PI parameters and caclulate PIs.
pers_image_ = gd.representations.PersistenceImage(bandwidth = 0.05 * max_PDs_temp, 
                                                  weight = lambda x: x[1]**2, resolution = [10, 10])
PIs_temp = pers_image_.fit_transform(PDs_temp)


    
# Print figure of images for Table 1 in turkevs2021noise.
num_fig_rows = 1
num_fig_cols = data_temp.shape[0]
subfig_height = 1
subfig_width = 1
fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height)) 
fig.tight_layout(pad = 0) 
for column, data_point in enumerate(data_temp):
    plot_image(data_point.reshape(num_x_pixels, num_y_pixels), vmin = min_data, vmax = max_data, axes = axes[column])

# Plot PDs, PLs, PIs.
num_fig_rows = 5
num_fig_cols = data_temp.shape[0]
subfig_height = 2
subfig_width = 2
fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height)) 
fig.tight_layout(pad = 0.5) 
for column, data_point in enumerate(data_temp):
    # Image.
    image = np.copy(data_temp[column])
    image = image.reshape((num_x_pixels, num_y_pixels))
    plot_image(image, vmin = np.min(data_temp), vmax = np.max(data_temp), axes = axes[0, column])
    axes[0, column].set_title("image %d" %(column+1), fontsize = 15)
    # Filtration function values.
    filt_func_vals = np.copy(filt_func_vals_temp[column])
    filt_func_vals = filt_func_vals.reshape((num_x_pixels, num_y_pixels))
    plot_image(filt_func_vals, vmin = 0, vmax = np.max(filt_func_vals_temp), axes = axes[1, column]) 
    axes[1, column].set_title(filt_temp + " filtration \n function", fontsize = 15)
    # PD.
    PD = np.copy(PDs_temp[column])
    plot_PD(PD, xymax = max_PDs_temp, axes = axes[2, column]) 
    axes[2, column].set_title("PD", fontsize = 15)   
    # PL.
    PL = np.copy(PLs_temp[column])
    plot_PL(PL, num_lndscs = 10, lndsc_resolution = 100, ymax = np.max(PLs_temp), axes = axes[3, column])
    axes[3, column].set_title("PL", fontsize = 15)
    # PI.
    PI = np.copy(PIs_temp[column])
    plot_PI(PI, pers_image_resolution = 10, vmin = 0, vmax = np.max(PIs_temp), axes = axes[4, column])
    axes[4, column].set_title("PI", fontsize = 15)     
plt.show()
