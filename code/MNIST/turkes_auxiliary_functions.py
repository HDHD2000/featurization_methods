##AUXILIARY FUNCTIONS FROM TURKES' NOTEBOOK


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

def affine_transform(data, transform_params):
    '''
    Calculate the image under given affine transformation, for each image in the dataset.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    transform_params: parameter for keras.preprocessing.image.ImageDataGenerator which defines the transformation,
    'theta': Float. Rotation angle in degrees.
    'tx': Float. Shift in the x direction.
    'ty': Float. Shift in the y direction.
    'shear': Float. Shear angle in degrees.
    'zx': Float. Zoom in the x direction.
    'zy': Float. Zoom in the y direction.abs
    'flip_horizontal': Boolean. Horizontal flip.
    'flip_vertical': Boolean. Vertical flip.
    'channel_shift_intensity': Float. Channel shift intensity.
    'brightness': Float. Brightness shift intensity.
    e.g., transform_parameters = {'theta':40, 'brightness':0.8, 'zx':0.9, 'zy':0.9}
    
    Output:
    data_trnsf: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values 
    of an image under transformation.
    '''
    num_data_points = data.shape[0] 
    num_pixels = data.shape[1] 
    num_x_pixels = np.sqrt(num_pixels).astype(int)
    num_y_pixels = num_x_pixels   
    data_images = data.reshape((num_data_points, num_x_pixels, num_y_pixels))    
    data_trnsf = np.zeros((num_data_points, num_pixels))    
    datagen = ImageDataGenerator()
    for p, image in enumerate(data_images):
        # We first need to represent each image as a num_x_pixels x num_y_pixels x 1 numpy array.
        image_3D_array = np.zeros((num_x_pixels, num_y_pixels, 1))
        for i in range(num_x_pixels):
            for j in range(num_x_pixels):
                image_3D_array[i][j] = [image[i][j]]
        # Next, we transform the image using keras.preprocessing.image import ImageDataGenerator.
        image_trnsf_3D_array = datagen.apply_transform(x = image_3D_array, transform_parameters = transform_params)
        # Finally, we represent the transformed image as a vector of length num_x_pixels x num_y_pixels.
        image_trnsf = np.zeros((num_x_pixels, num_y_pixels))
        for i in range(num_x_pixels):
            for j in range(num_y_pixels):
                image_trnsf[i][j] = image_trnsf_3D_array[i][j].item()
        data_point_trnsf = image_trnsf.reshape(num_pixels, )
        data_trnsf[p] = data_point_trnsf
    return data_trnsf


def gaussian_noise(data, noise_std_dev):
    '''
    Calculate the image under gaussian noise, for each image in the dataset.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    noise_std_dev: a float, standard deviation of gaussian noise. Most of the values are within +-3std_dev from the mean 0.
    
    Output:
    data_gaussian_noise: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values 
    of an image under gaussian noise.
    '''
    data_gaussian_noise = np.zeros(data.shape)    
    for p, data_point in enumerate(data):
        # We apply noise to each image separately, as we want have noise_mean=0 and noise_std_dev for each image (row in data)
        # and not for this to be the mean and std dev for the whole 2D matrix data.
        gaussian_noise = np.random.normal(0, noise_std_dev, data_point.shape)
        data_point_gaussian_noise = data_point + gaussian_noise
        data_gaussian_noise[p] = data_point_gaussian_noise
    data_gaussian_noise = np.clip(data_gaussian_noise, np.min(data), np.max(data))
    return data_gaussian_noise


def salt_and_pepper_noise(data, proportion_noise_pixels):
    '''
    Calculate the image under salt and pepper noise, for each image in the dataset. 
    Salt and pepper noise changes the value of a given ratio of pixels, which are selected randomly, 
    to either salt (white) or pepper (black).
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    proportion_noise_pixels: a float, proportion of pixels to be changed.
    
    Output:
    data_sp_noise: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values 
    of an image under salt and pepper noise.    
    '''
    data_sp_noise = np.zeros(data.shape)    
    for p, data_point in enumerate(data):
        # Each pixel changes (pixel_chaned[i, j] = 1) with a given probability.
        pixel_changed = np.random.binomial(1, proportion_noise_pixels, data_point.shape)
        # The pixels which change, change equally likely to salt (white) or peper (black).      
        # For the MNIST images, it might be more reasonable to set the probability of changing to pepper as very large 
        # (e.g., 0.8), since it is only then that something changes for white pixels, which are in the large majority.
        sp_noise = np.random.binomial(1, 0.5, data_point.shape) * np.max(data)
        # if pixel_chaned[i][j] = 0
        # data_point_image_sp_noise[i][j] = 1 * data_point_image[i][j] + 0 * sp_noise = data_point_image[i][j]
        # if pixel_chaned[i][j] = 1
        # data_point_image_sp_noise[i][j] = 0 * data_point_image[i][j] + 1 * sp_noise = sp_noise[i][j]     
        data_point_sp_noise = (1 - pixel_changed)*data_point + pixel_changed*sp_noise       
        data_sp_noise[p] = data_point_sp_noise
    return data_sp_noise  
    
    
def shot_noise(data, pixel_scaling_factor):
    '''
    Calculate the image under shot noise, for each image in the dataset. 
    Shot noise has a root-mean-square value proportional to the square root of the image intensity, 
    and the noise at different pixels are independent of one another. Shot noise follows a Poisson distribution, 
    which except at very high intensity levels approximates a Gaussian distribution of only positive numbers.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    pixel_scaling_factor: a float, level of noise.
    
    Output:
    data_shot_noise: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values 
    of an image under shot noise.    
    '''
    data_shot_noise = np.zeros(data.shape)   
    pixel_scaling_factor = 1 / pixel_scaling_factor 
    for p, data_point in enumerate(data):
        # The Poisson noise is signal dependent; it is not additive as Gaussian noise. 
        # Each new pixel is defined as a value which is drawn from a "normal" positive (i.e., Poisson) distribution
        # with distribution mean being the original pixel value. 
        data_point_shot_noise = np.random.poisson(data_point * pixel_scaling_factor) / float(pixel_scaling_factor) 
        data_shot_noise[p] = data_point_shot_noise
    data_shot_noise = np.clip(data_shot_noise, np.min(data), np.max(data))   
    return data_shot_noise

##---------------------------------------------##

def binary_filtration_function(data, threshold_grsc_perc):
    '''
    Calculate the binary filtration function values for each image in the dataset. 
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    threshold_grsc_perc: a float in [0,1], representing the threshold greyscale value percentage to obtain the binary image.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    binary filtration function on each pixel of an image.
    '''
    num_data_points = data.shape[0]
    num_pixels = data.shape[1]
    filt_func_vals_data = np.zeros((num_data_points, num_pixels))
    for p, data_point in enumerate(data): 
        filt_func_vals_data[p] = data_point > threshold_grsc_perc * np.max(data_point)
        # Pixels with the lowest filtration function values appear first in the filtration. 
        filt_func_vals_data[p] = 1 - filt_func_vals_data[p]
    return filt_func_vals_data    


def greyscale_filtration_function(data):
    '''
    Calculate the greyscale filtration function values for each image in the dataset. 
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    greyscale filtration function on each pixel of an image.
    '''
    num_data_points = data.shape[0]
    num_pixels = data.shape[1]
    filt_func_vals_data = np.zeros((num_data_points, num_pixels))
    for p, data_point in enumerate(data):     
        # Pixels with the lowest filtration function values appear first in the filtration. 
        filt_func_vals_data[p] = np.max(data_point) - data_point        
    return filt_func_vals_data    


def density_filtration_function(data, threshold_grsc_perc, max_dist):  
    '''
    Calculate the density filtration function values for each image in the dataset. 
    Density filtration function counts the number of dark-enough pixels in an image, within a given distance.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    threshold_grsc_perc: a float in [0,1], representing the threshold greyscale value percentage to obtain the binary image.
    max_dist: a non-negative integer, representing the size of the considered neighborhood for each pixel in an image.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    density filtration function on each pixel of an image.
    '''
    num_data_points = data.shape[0]
    num_x_pixels = np.sqrt(data.shape[1]).astype(int)
    num_y_pixels = num_x_pixels
    filt_func_vals_data = np.zeros((num_data_points, num_x_pixels * num_y_pixels))    
    point_cloud_complete = np.zeros((num_x_pixels * num_y_pixels, 2))
    p = 0
    for i in range(num_x_pixels):
        for j in range(num_y_pixels):
            point_cloud_complete[p, 0] = j
            point_cloud_complete[p, 1] = num_y_pixels - i
            p = p + 1            
    for p, data_point in enumerate(data): 
        point_cloud = build_point_cloud(data_point, threshold_grsc_perc)        
        kdt = KDTree(point_cloud, leaf_size = 30, metric = "euclidean") 
        num_nbhs = kdt.query_radius(point_cloud_complete, r = max_dist, count_only = True)
        filt_func_vals = num_nbhs
        max_num_nbhs = 2 * max_dist**2 + 2 * max_dist + 1 # num of pixels in euclidean ball with radius max_dist
        filt_func_vals = max_num_nbhs - filt_func_vals
        # Cut-off density value (so that density does not also reflect the size of the hole).
        # max_filt_func_val = np.max(filt_func_vals)
        # filt_func_vals[filt_func_vals > 0.65 * max_filt_func_val] = max_filt_func_val # 0.4 in carriere2018statistical  
        filt_func_vals_data[p] = filt_func_vals         
    return filt_func_vals_data


def radial_filtration_function(data, threshold_grsc_perc, x_pixel, y_pixel):  
    '''
    Calculate the radial filtration function values for each image in the dataset.
    Radial filtration function corresponds to the distance from a given reference pixel.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    threshold_grsc_perc: a float in [0,1], representing the threshold greyscale value percentage to obtain the binary image.
    x_pixel, y_pixel: integers, coordinates of the reference pixel.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    radial filtration function on each pixel of an image.
    '''
    num_data_points = data.shape[0]
    num_x_pixels = np.sqrt(data.shape[1]).astype(int)
    num_y_pixels = num_x_pixels    
    filt_func_vals_data = np.zeros((num_data_points, num_x_pixels * num_y_pixels))
    for p, data_point in enumerate(data):                   
        image = data_point.reshape((num_x_pixels, num_y_pixels))    
        binary_image = image >= threshold_grsc_perc * np.max(image)         
        filt_func_vals = np.zeros((num_x_pixels, num_y_pixels))
        point_reference = np.array([x_pixel, y_pixel])
        for i in range(num_x_pixels):
            for j in range(num_y_pixels):
                point = np.array([i, j])
                filt_func_vals[i, j] = np.linalg.norm(point - point_reference, ord = 2)  
                # filt_func_vals[i, j] = np.inner(point, point_reference)   
        filt_func_vals[binary_image == 0] = np.max(filt_func_vals)                    
        filt_func_vals_data[p] = filt_func_vals.reshape((num_x_pixels * num_y_pixels, ))                 
    return filt_func_vals_data


def build_point_cloud(data_point, threshold_grsc_perc): 
    '''
    Calculate the point cloud from an image, which sees a point cloud point for any pixel with the
    greyscale value above the given threshold.
    
    Input:
    data_point: a num_pixels x 1 numpy array, representing the greyscale pixel values of an image.
    threshold_grsc_perc: a float in [0,1], representing the threshold greyscale value percentage to obtain the binary image.
    
    Output:
    point_cloud: a num_black_pixels_data_point x 2 numpy array, x and y coordinates of each point cloud point.
    '''    
    num_x_pixels = np.sqrt(data_point.shape[0]).astype(int)
    num_y_pixels = num_x_pixels     
    image = data_point.reshape((num_x_pixels, num_y_pixels))    
    binary_image = image >= threshold_grsc_perc * np.max(image)          
    num_black_pixels = np.sum(binary_image) 
    point_cloud = np.zeros((num_black_pixels, 2))
    point = 0        
    for i in range(num_x_pixels):
        for j in range(num_y_pixels):
            if binary_image[i, j] > 0:
                point_cloud[point, 0] = j
                point_cloud[point, 1] = num_y_pixels - i
                point = point + 1        
    return point_cloud


def distance_filtration_function(data, threshold_grsc_perc):
    '''
    Calculate the Rips filtration (i.e., empirical distance) function values for a point cloud, obtained from each image in the dataset. 
    Rips filtration function corresponds to the distance to the closest dark-enough pixel.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    threshold_grsc_perc: a float in [0,1], representing the threshold greyscale value percentage to obtain the binary image.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    Rips filtration function on discretized R^d.
    '''      
    return dtm_filtration_function(data, threshold_grsc_perc, 0)


def dtm_filtration_function(data, threshold_grsc_perc, m):          
    '''
    Compute the DTM filtration function values (with exponent p=2) of the empirical measure of a point cloud, obtained from each image in the dataset. 
    DTM filtration function corresponds to the average distance to some of the closest dark-enough pixels.
    This is an adaption from DTM implementation available at: 
    https://github.com/GUDHI/TDA-tutorial/blob/master/DTM_filtrations.py (from Anai et al, DTM-based filtrations).
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    threshold_grsc_perc: a float in [0,1], representing the threshold greyscale value percentage to obtain the binary image.
    m: a float in [0, 1] reflecting the number of neighbors to be considered for each point in point cloud.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    DTM filtration function on discretized R^d.
    '''      
    num_data_points = data.shape[0]
    num_x_pixels = np.sqrt(data.shape[1]).astype(int)
    num_y_pixels = num_x_pixels    
    filt_func_vals_data = np.zeros((num_data_points, num_x_pixels * num_y_pixels))    
    point_cloud_complete = np.zeros((num_x_pixels * num_y_pixels, 2))
    p = 0
    for i in range(num_x_pixels):
        for j in range(num_y_pixels):
            point_cloud_complete[p, 0] = j
            point_cloud_complete[p, 1] = num_y_pixels - i
            p = p + 1    
    for p, data_point in enumerate(data):    
        point_cloud = build_point_cloud(data_point, threshold_grsc_perc)        
        kdt = KDTree(point_cloud, leaf_size = 30, metric = "euclidean")         
        num_points = point_cloud.shape[0]
        num_nbhs = math.floor(m * num_points) + 1
        dists_to_nbhs, _ = kdt.query(point_cloud_complete, num_nbhs, return_distance = True)  
        filt_func_vals = np.sqrt(np.sum(dists_to_nbhs*dists_to_nbhs, axis = 1) / num_nbhs) 
        # Cut-off DTM value, as in carriere2018statistical. No, because in this case DTM does not capture size of hole.
        # max_filt_func_val = np.max(filt_func_vals)
        # filt_func_vals[filt_func_vals > 0.5 * max_filt_func_val] = max_filt_func_val
        filt_func_vals_data[p] = filt_func_vals  
    return filt_func_vals_data


##----------------------------------------------##

def pers_intervals_across_homdims(filt_func_vals_data, filt = " ", data = [], threshold_grsc_perc = 0.5):
    '''
    Compute persistent homology with respect to the given filtration for each filtration in the dataset.   
    
    Input:
    > filt_func_vals_data: a num_data_points x num_pixels numpy array, with each row representing the filtration function 
    values on each pixel/cell in discretized R^2.
    > filt: a string that determines the choice of filtration.
    
    Persistent homology is calculated on a filtration, but for the PH calculated with respect to simplicial complexes built on point clouds, 
    GUDHI implementation requires the point cloud itself, in order to calculate the distances between point cloud points which determine 
    when an edge appears in the filtration. The additional parameters that are needed for the calculation of persistent homology
    on point clouds are the following:    
    > data: a num_data_points x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    > threshold_grsc_perc: a float in [0, 1], representing the threshold greyscale value percentage used to build a point cloud from image.
    
    Output:
    > pers_intervals_homdim0_data: list of persistence intervals (b, d) corresponding to 0-dim cycles (connected components).
    > pers_intervals_homdim1_data: list of persistence intervals (b, d) corresponding to 1-dim cycles (holes).
    '''    
    pers_intervals_homdim0_data = []
    pers_intervals_homdim1_data = []   

    for i, filt_func_vals_data_point in enumerate(filt_func_vals_data):   
        
        if(filt == "Rips"):
            data_point = data[i]
            point_cloud = build_point_cloud(data_point, threshold_grsc_perc)
            simplicial_complex = gd.RipsComplex(points = point_cloud, max_edge_length = np.inf)
            simplex_tree = simplicial_complex.create_simplex_tree(max_dimension = 2)  
        
        elif(filt == "DTM"):          
            data_point = data[i]
            point_cloud = build_point_cloud(data_point, threshold_grsc_perc)               
            # To build a weighted simplex tree, we need:
            # 1) distance matrix between point cloud points
            distance_matrix = euclidean_distances(point_cloud)
            # 2) (one half of) weight, i.e., filtration function value for each point cloud point
            num_points = point_cloud.shape[0]
            filt_func_vals_point_cloud = np.zeros(num_points)
            num_x_pixels = np.sqrt(filt_func_vals_data.shape[1]).astype(int) 
            num_y_pixels = num_x_pixels
            filt_func_vals_data_point = np.copy(filt_func_vals_data_point).reshape((num_x_pixels, num_y_pixels))            
            # In build point cloud, image -> point cloud: x = j, y = num_y_pixels - i,
            # so that an inverse transformation is necessary here: j = x, i = num_y_pixels - y.                
            for p, point in enumerate(point_cloud):  
                i = num_y_pixels - point[1].astype(int)
                j = point[0].astype(int)
                filt_func_vals_point_cloud[p] = filt_func_vals_data_point[i, j]               
            simplicial_complex = WeightedRipsComplex(distance_matrix = distance_matrix, weights = filt_func_vals_point_cloud, max_filtration = np.inf)  
            simplex_tree = simplicial_complex.create_simplex_tree(max_dimension = 2)
        
        else:                       
            num_x_pixels = np.sqrt(filt_func_vals_data.shape[1]).astype(int) 
            num_y_pixels = num_x_pixels
            simplicial_complex = gd.CubicalComplex(dimensions = [num_x_pixels, num_y_pixels], 
                                                   top_dimensional_cells = filt_func_vals_data_point) 
            simplex_tree = simplicial_complex # CubicalComplex is similar to simplex tree

        homdims_pers_intervals = simplex_tree.persistence() # needs to be called before persistence_intervals_in_dimension()           
        pers_intervals_homdim0 = simplex_tree.persistence_intervals_in_dimension(0)
        pers_intervals_homdim1 = simplex_tree.persistence_intervals_in_dimension(1)        
       
        # If the list of persistence intervals is empty, gd.representations.PersistenceImage().fit_transfrom()
        # returns an error. Therefore, we replace such lists with [[0, 0]], and this has no influence on the 
        # calculation of Wasserstein distances.
        if(len(pers_intervals_homdim0) == 0):
            pers_intervals_homdim0 = np.asarray([[0, 0]])
        if(len(pers_intervals_homdim1) == 0):
            pers_intervals_homdim1 = np.asarray([[0, 0]])    

        pers_intervals_homdim0_data.append(pers_intervals_homdim0)
        pers_intervals_homdim1_data.append(pers_intervals_homdim1)

    return pers_intervals_homdim0_data, pers_intervals_homdim1_data


##-------------------------------------------##

def l_p_distances(data_1, data_2, p):
    '''
    Calculate l_p distance between an image in one dataset, and the corresponding image in the other dataset.
    
    Input:
    > data_1, data_2: a num_data_points x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    > p: a float in [1, \infty], reflecting the choice of l_p metric.
    
    Output:
    > dists_data_1_2: a num_data_points x 1 array, with each row representing the the distance between an image in data_1
    and the corresponding image (in the same row) in data_2.
    ''' 
    num_data_points = data_1.shape[0]
    dists_data_1_2 = np.zeros((num_data_points, ))
    for i in range(num_data_points):
        dists_data_1_2[i] = np.linalg.norm(data_1[i] - data_2[i], ord = p)
    return dists_data_1_2


def wasserstein_p_q_distances(pers_intervals_homdim_data_1, pers_intervals_homdim_data_2, p, q):
    '''
    Calculate Wasserstein_p_g distance between an persistence diagram in one dataset, and the corresponding 
    persistence diagram in the other dataset.
    
    Input:
    > pers_intervals_homdim_data_1, pers_intervals_homdim_data_2: lists of num_data_points elements, 
    each a num_cycles x 2 numpy array of birth and death values for each cycle in the given homological dimension
    (a persistence diagram).
    > p, q: floats in [1, \infty], reflecting the choice of Wasserstein_p_q metric.
    
    Output:
    > dists_pers_intervals_homdim_data_1_2: a num_data_points x 1 array, with each row representing the the distance 
    between a persistence diagram in pers_intervals_homdim_data_1 and the corresponding persistence diagram (in the same row)
    in pers_intervals_homdim_data_2.
    ''' 
    num_data_points = len(pers_intervals_homdim_data_1)
    dist_pers_intervals_homdim_data_1_2 = np.zeros((num_data_points, ))
    for i in range(num_data_points):
        if p == np.inf:
            dist_pers_intervals_homdim_data_1_2[i] = gd.bottleneck_distance(pers_intervals_homdim_data_1[i], 
                                                                            pers_intervals_homdim_data_2[i], e = 0)
        else:
            dist_pers_intervals_homdim_data_1_2[i] = gudhi.wasserstein.wasserstein_distance(pers_intervals_homdim_data_1[i], 
                                                                                            pers_intervals_homdim_data_2[i], 
                                                                                            order = p, internal_p = q)
    return dist_pers_intervals_homdim_data_1_2


@njit(parallel = True)
def l_p_distance_matrix(data, p):
    '''
    Calculate l_p distance between any two images in the given dataset.
    
    Input:
    > data: a num_data_points x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    > p: a float in [1, \infty], reflecting the choice of l_p metric.
    
    Output:
    > distance_matrix_data: a num_data_points x num_data_point numpy array, with each element corresponding to the distance
    between two images.    
    ''' 
    num_data_points = len(data)
    distance_matrix_data = np.zeros((num_data_points, num_data_points))
    for i in prange(num_data_points):
        for j in prange(i):
            distance_matrix_data[i, j] = np.linalg.norm(data[i] - data[j], ord = p)
    # Distance matrix elements above the diagonal.
    distance_matrix_data =  distance_matrix_data + distance_matrix_data.transpose()
    return distance_matrix_data


def wasserstein_p_q_distance_matrix(pers_intervals_homdim, p, q):
    '''
    Calculate the Wasserstein W_p_q distance between any two persistence diagrams in the given dataset.
    
    Input:
    > pers_intervals_homdim: a list of num_data_points elements, each a num_cycles x 2 numpy array of birth and death
    values for each cycle in the given homological dimension.
    > p, q: floats in [1, \infty], reflecting the choice of Wasserstein_p_q metric.
    
    Output:
    > distance_matrix_data: a num_data_points x num_data_point numpy array, with each element corresponding to the distance
    between two persistence diagrams.    
    ''' 
    num_data_points = len(pers_intervals_homdim)
    distance_matrix_homdim = np.zeros((num_data_points, num_data_points))
    for i in range(num_data_points):
        for j in range(i):
            if p == np.inf:
                distance_matrix_homdim[i, j] = gd.bottleneck_distance(pers_intervals_homdim[i], pers_intervals_homdim[j], e = 0) # e=0?
            else:
                distance_matrix_homdim[i, j] = gudhi.wasserstein.wasserstein_distance(pers_intervals_homdim[i], 
                                                                                      pers_intervals_homdim[j], 
                                                                                      order = p, internal_p = q)
    # Distance matrix elements above the diagonal.
    distance_matrix_homdim = distance_matrix_homdim + distance_matrix_homdim.transpose()
    return distance_matrix_homdim


##---------------------------------------------##

def plot_image(image, vmin, vmax, axes):
    '''
    Plot the given image.
    
    Input:
    > image: a num_x_pixels x num_y_pixels numpy array.
    > vmin, vmax: mininum and maximum pixel value.
    > axes: an object of class Axes, axes to plot the image.  
    ''' 
    axes.matshow(image, cmap = plt.cm.gray_r, vmin = vmin, vmax = vmax)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xticklabels([])
    axes.set_yticklabels([])

    
def plot_PD(PD, xymax, axes):    
    '''
    Plot the given persistence diagram (PD).
    
    Input:
    > PD: a num_cycles x 2 numpy array of birth and death values for each cycle, i.e., a persistence diagram (PD).
    > xymax: a positive float, maximum value of the plot x and y axis.
    > axes: an object of class Axes, axes to plot the PD.  
    ''' 
    axes.scatter(PD[:, 0], PD[:, 1], 80, c = "green")
    axes.set_aspect(aspect = 'equal')
    axes.set_xlim(-0.05 * xymax, 1.1 * xymax)
    axes.set_ylim(-0.05 * xymax, 1.1 * xymax)
    x = np.arange(-0.05 * xymax, 1.1 * xymax, 0.01)
    axes.plot(x, x, c = 'black') # plot the diagonal    
    intervals_unique, multiplicities = np.unique(PD, axis = 0, return_counts = True) 
    for i, multiplicity in enumerate(multiplicities):
        if multiplicity > 1: # Annotate a persistence interval/PD point only if it appears multiple times, for visual clarity.
            axes.annotate(multiplicity, xy = (intervals_unique[i, 0], intervals_unique[i, 1]), ha = "left", va = "bottom",
                          xytext = (5, 0), textcoords = "offset points", fontsize = 15, color = "green")    
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    
    
def plot_PL(PL, num_lndscs, lndsc_resolution, ymax, axes):
    '''
    Plot the given persistence landscape (PL).
    
    Input:
    > PL: a num_lndscs x lndsc_resolution x 1 numpy array, a persistence landscape (PL).
    > num_lndscs: a natural number, number of different landscape functions.
    > lndsc_resolution: a natural number, number of equidistant point that the landscape function is evaluated on to obtain
    the vectorized persistence landscape.
    > ymax: a positive float, maximum value of the plot y-axis.
    > axes: an object of class Axes, axes to plot the PL.  
    ''' 
    cmap = plt.get_cmap('Dark2')
    for i in range(num_lndscs):
        axes.plot(PL[i*lndsc_resolution : (i+1)*lndsc_resolution], linewidth = 5, c = cmap.colors[i % 8]) # Only 8 colors in Dark2 colormap.
        axes.set_ylim(0, 1.1 * ymax) 
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xticklabels([])
    axes.set_yticklabels([])
        

def plot_PI(PI, pers_image_resolution, vmin, vmax, axes):
    '''
    Plot the given persistence image (PI).
    
    Input:
    > PI: a PI_resolution*PI_resolution x 1 numpy array, a persistence image (PI).
    > PI_resolution: a natural number, resolution of the persistence image.
    > lndsc_resolution: a natural number, number of equidistant point that the landscape function is evaluated on to obtain
    the vectorized persistence landscape.
    > vmin, vmax: mininum and maximum pixel value.
    > axes: an object of class Axes, axes to plot the image.  
    ''' 
    plot_image(np.flip(PI.reshape((pers_image_resolution, pers_image_resolution)), 0), vmin = vmin, vmax = vmax, axes = axes)


def plot_bar_containers(x_ticks_labels, y_values_per_bar_container, legend_labels):  
    '''
    Plot the grouped bar chart with given values.
    
    Input:
    > x_tick_labels: a list of string labels on the x-axis.
    > y_values_per_bar_container: a dictionary, with each element being a list of y-axis values for a specific 
    group/color/legend item.
    > legend labels: a list of string legend labels.
    ''' 
    num_x_ticks = len(x_ticks_labels) 
    x_ticks = np.arange(num_x_ticks)      
    num_bars_per_x_tick = len(y_values_per_bar_container) # = len(legend_labels)   
    width_bar = 0.65 * 1/num_bars_per_x_tick
    width_from_x_tick = np.arange(num_bars_per_x_tick) - np.floor(num_bars_per_x_tick/2)  # (..., -3, -2, -1, 0, 1, 2, 3, ...)
    fig, axes = plt.subplots(figsize = (20, 7)) 
    cmap = plt.get_cmap('tab20') # we have two levels of each type of noise
    bar_colors = ["black"] + [cmap.colors[t] for t in range(16)] # no-noise and two levels of 8 trnsfs     
    for t, legend_label in enumerate(legend_labels):    
        axes.bar(x_ticks + width_from_x_tick[t] * width_bar, y_values_per_bar_container[legend_label], width_bar, 
                 label = legend_label, color = bar_colors[t])        
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_ticks_labels, fontsize = 10)
    axes.set_axisbelow(True)
    return fig, axes


