from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
#from libsvm.svmutil import *
from sklearn.preprocessing import scale
import sys
import pandas as pd
import gudhi as gd
import time

from persistence_methods import swk_features, landscape_features, persistence_image_features, silhouette_features, pwgk_features, pssk_features, pfk_features, entropy_features, carlsson_coordinates, tropical_coordinates

##===================================================##
##Loading the MNIST data set using keras module

start = time.time()

(train_X, train_y), (test_X, test_y) = mnist.load_data()

##===================================================##

##function giving the four different directional sweeps of an image
def directional_transform(img):
    z = np.array([0,1])
    z = z.reshape([1,2])
    left_to_right = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                left_to_right[i,j] = np.inner([i,j],z)
            else:
                left_to_right[i,j] = 0

    z = np.array([0,-1])
    z = z.reshape([1,2])
    right_to_left = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                right_to_left[i,j] = abs(28 + np.inner([i,j],z))
            else:
                right_to_left[i,j] = 0

    z = np.array([1,0])
    z = z.reshape([1,2])
    bottom_to_top = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                bottom_to_top[i,j] = np.inner([i,j],z)
            else:
                bottom_to_top[i,j] = 0

    z = np.array([-1,0])
    z = z.reshape([1,2])
    top_to_bottom = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if img[i,j] != 0:
                top_to_bottom[i,j] = abs(28 + np.inner([i,j],z))
            else:
                top_to_bottom[i,j] = 0
    imgs = [left_to_right, right_to_left, bottom_to_top, top_to_bottom]
    return imgs


##Function computing the persistence diagrams giving the four sweeps of an image
def compute_persistence(directional_transform):
    simplex_tree = gd.RipsComplex(directional_transform).create_simplex_tree(max_dimension = 2)
    barcode = simplex_tree.persistence()
    dgms_lower_0 = simplex_tree.persistence_intervals_in_dimension(0)
    dgms_lower_1 = simplex_tree.persistence_intervals_in_dimension(1)
    if len(dgms_lower_1) == 0:
        dgms_lower_1 = np.array([0.0,0.0]).reshape((1,2))
    dgms_lower = [dgms_lower_0, dgms_lower_1]
    return dgms_lower


##appends persistence diagrams (dgms) to a list of persistence diagrams (dim_list)
def append_dim_list(dgms, dim_list):
    jth_pt = []
    for k in range(0, len(dgms)):
        if dgms[k][1] - dgms[k][0] >=0:
            birth = dgms[k][0]
            death = dgms[k][1]
        else:
            birth = dgms[k][1]
            death = dgms[k][0]
        if math.isinf(death):
            b = 100
        else:
            b = death
        t = [birth, b]
        jth_pt.append(t)
    dim_list.append(np.array(jth_pt))
    
##=============================================================##

#COMPUTING persistence diagrams

##Select the training and testing set

num_images = 1000

tuning = np.arange(0,num_images)
tuning_index = tuning
n = tuning_index.shape[0]
obs = np.arange(0,n)
zero_dim_0 = []
zero_dim_1 = []
zero_dim_2 = []
zero_dim_3 = []
one_dim_0 = []
one_dim_1 = []
one_dim_2 = []
one_dim_3 = []

##forming the directional sweeps and the corresponding persistence diagrams
for i in tuning_index:
    img = test_X[i]
    imgs = directional_transform(img)
    for j in range(0,4):
        dgms_lower= compute_persistence(imgs[j])
        if j == 0:
            append_dim_list(dgms_lower[0], zero_dim_0)
            append_dim_list(dgms_lower[1], one_dim_0)
        if j == 1:
            append_dim_list(dgms_lower[0], zero_dim_1)
            append_dim_list(dgms_lower[1], one_dim_1)
        if j == 2:
            append_dim_list(dgms_lower[0], zero_dim_2)
            append_dim_list(dgms_lower[1], one_dim_2)
        if j == 3:
            append_dim_list(dgms_lower[0], zero_dim_3)
            append_dim_list(dgms_lower[1], one_dim_3) 
            
train_index, test_index, original_index_train, original_index_test, y_train, y_test = train_test_split(obs, tuning_index, test_y[tuning_index], test_size = .3, random_state=1, stratify = train_y[tuning_index])

zero_dim_ltr_train = np.array(zero_dim_0)[train_index]
zero_dim_rtl_train = np.array(zero_dim_1)[train_index]
zero_dim_ttb_train = np.array(zero_dim_2)[train_index]
zero_dim_btt_train = np.array(zero_dim_3)[train_index]

zero_dim_ltr_test = np.array(zero_dim_0)[test_index]
zero_dim_rtl_test = np.array(zero_dim_1)[test_index]
zero_dim_ttb_test = np.array(zero_dim_2)[test_index]
zero_dim_btt_test = np.array(zero_dim_3)[test_index]


one_dim_ltr_train = np.array(one_dim_0)[train_index]
one_dim_rtl_train = np.array(one_dim_1)[train_index]
one_dim_ttb_train = np.array(one_dim_2)[train_index]
one_dim_btt_train = np.array(one_dim_3)[train_index]

one_dim_ltr_test = np.array(one_dim_0)[test_index]
one_dim_rtl_test = np.array(one_dim_1)[test_index]
one_dim_ttb_test = np.array(one_dim_2)[test_index]
one_dim_btt_test = np.array(one_dim_3)[test_index]

dgms_lower = compute_persistence(imgs[0])

##===================================================##
##LANDSCAPE METHOD
 
"""
n = [20]
r = [500]

train_accuracy = []
test_accuracy = []
n_model = []
r_model = []
c_model = []
model_type = []

for i in n:
    for j in r:
        X_train_features_1_ltr_landscapes, X_test_features_1_ltr_landscapes = landscape_features(one_dim_ltr_train, one_dim_ltr_test, num_landscapes=i, resolution=j)
        X_train_features_0_ltr_landscapes, X_test_features_0_ltr_landscapes = landscape_features(zero_dim_ltr_train, zero_dim_ltr_test, num_landscapes=i, resolution=j)

        X_train_features_1_rtl_landscapes, X_test_features_1_rtl_landscapes = landscape_features(one_dim_rtl_train, one_dim_rtl_test, num_landscapes=i, resolution=j)
        X_train_features_0_rtl_landscapes, X_test_features_0_rtl_landscapes = landscape_features(zero_dim_rtl_train, zero_dim_rtl_test, num_landscapes=i, resolution=j)

        X_train_features_1_ttb_landscapes, X_test_features_1_ttb_landscapes = landscape_features(one_dim_ttb_train, one_dim_ttb_test, num_landscapes=i, resolution=j)
        X_train_features_0_ttb_landscapes, X_test_features_0_ttb_landscapes = landscape_features(zero_dim_ttb_train, zero_dim_ttb_test, num_landscapes=i, resolution=j)

        X_train_features_1_btt_landscapes, X_test_features_1_btt_landscapes = landscape_features(one_dim_btt_train, one_dim_btt_test, num_landscapes=i, resolution=j)
        X_train_features_0_btt_landscapes, X_test_features_0_btt_landscapes = landscape_features(zero_dim_btt_train, zero_dim_btt_test, num_landscapes=i, resolution=j)
        
        X_train_features = np.column_stack((X_train_features_1_ltr_landscapes,X_train_features_1_rtl_landscapes,X_train_features_1_ttb_landscapes,X_train_features_1_btt_landscapes,X_train_features_0_ltr_landscapes,X_train_features_0_rtl_landscapes,X_train_features_0_btt_landscapes,X_train_features_0_ttb_landscapes))
        X_test_features = np.column_stack((X_test_features_1_ltr_landscapes,X_test_features_1_rtl_landscapes,X_test_features_1_ttb_landscapes,X_test_features_1_btt_landscapes,X_test_features_0_ltr_landscapes,X_test_features_0_rtl_landscapes,X_test_features_0_btt_landscapes,X_test_features_0_ttb_landscapes))

        c = [80]
        for k in c:
            clf = SVC(kernel='rbf', C=k).fit(X_train_features, y_train)
            train_accuracy.append(clf.score(X_train_features, y_train))
            test_accuracy.append(clf.score(X_test_features, y_test))
            n_model.append(i)
            r_model.append(j)
            model_type.append('SVC')
            c_model.append(k)

landscape_results = pd.DataFrame()
landscape_results['Training Accuracy'] = train_accuracy
landscape_results['Test Accuracy'] = test_accuracy
landscape_results['n'] = n_model
landscape_results['r'] = r_model
landscape_results['c'] = c_model
landscape_results['Model Type'] = model_type

landscape_sorted = landscape_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(landscape_sorted[0:50])
"""

##========================================================##

##PERSISTENCE IMAGE METHOD

"""
pixels = [[30,30]]
bandwidth = [10]

train_accuracy = []
test_accuracy = []
p_model = []
s_model = []
c_model = []
model_type = []

for p in pixels:
    for s in bandwidth:
        X_train_features_0_ltr_imgs, X_test_features_0_ltr_imgs = persistence_image_features(zero_dim_ltr_train, zero_dim_ltr_test, pixels=p, bandwidth=s)
        X_train_features_0_rtl_imgs, X_test_features_0_rtl_imgs = persistence_image_features(zero_dim_rtl_train, zero_dim_rtl_test, pixels=p, bandwidth=s)
        X_train_features_0_ttb_imgs, X_test_features_0_ttb_imgs = persistence_image_features(zero_dim_ttb_train, zero_dim_ttb_test, pixels=p, bandwidth=s)
        X_train_features_0_btt_imgs, X_test_features_0_btt_imgs = persistence_image_features(zero_dim_btt_train, zero_dim_btt_test, pixels=p, bandwidth=s)

        X_train_features_1_ltr_imgs, X_test_features_1_ltr_imgs = persistence_image_features(one_dim_ltr_train, one_dim_ltr_test, pixels=p, bandwidth=s)
        X_train_features_1_rtl_imgs, X_test_features_1_rtl_imgs = persistence_image_features(one_dim_rtl_train, one_dim_rtl_test, pixels=p, bandwidth=s)
        X_train_features_1_ttb_imgs, X_test_features_1_ttb_imgs = persistence_image_features(one_dim_ttb_train, one_dim_ttb_test, pixels=p, bandwidth=s)
        X_train_features_1_btt_imgs, X_test_features_1_btt_imgs = persistence_image_features(one_dim_btt_train, one_dim_btt_test, pixels=p, bandwidth=s)
        
        X_train_features = np.column_stack((X_train_features_1_ltr_imgs,X_train_features_1_rtl_imgs,X_train_features_1_ttb_imgs,X_train_features_1_btt_imgs,X_train_features_0_ltr_imgs,X_train_features_0_rtl_imgs,X_train_features_0_btt_imgs,X_train_features_0_ttb_imgs))
        X_test_features = np.column_stack((X_test_features_1_ltr_imgs,X_test_features_1_rtl_imgs,X_test_features_1_ttb_imgs,X_test_features_1_btt_imgs,X_test_features_0_ltr_imgs,X_test_features_0_rtl_imgs,X_test_features_0_btt_imgs,X_test_features_0_ttb_imgs))

        c = [50]
        for i in c:
            clf = SVC(kernel='rbf', C=i).fit(X_train_features, y_train)
            train_accuracy.append(clf.score(X_train_features, y_train))
            test_accuracy.append(clf.score(X_test_features, y_test))
            p_model.append(p)
            s_model.append(s)
            model_type.append('SVC')
            c_model.append(i)
            
pi_results = pd.DataFrame()
pi_results['Training Accuracy'] = train_accuracy
pi_results['Test Accuracy'] = test_accuracy
pi_results['p'] = p_model
pi_results['s'] = s_model
pi_results['c'] = c_model
pi_results['Model Type'] = model_type

pi_sorted = pi_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(pi_sorted[0:60])

"""

##========================================================##

##SLICED WASSERSTEIN KERNEL METHOD

"""

num_directions = [10]
bandwidth = [10000]

train_accuracy = []
test_accuracy = []
p_model = []
s_model = []
c_model = []
model_type = []

for p in num_directions:
    for s in bandwidth:
        X_train_features_0_ltr_swk, X_test_features_0_ltr_swk = swk_features(zero_dim_ltr_train, zero_dim_ltr_test, num_direc=p, bandwidth=s)
        X_train_features_0_rtl_swk, X_test_features_0_rtl_swk = swk_features(zero_dim_rtl_train, zero_dim_rtl_test, num_direc=p, bandwidth=s)
        X_train_features_0_ttb_swk, X_test_features_0_ttb_swk = swk_features(zero_dim_ttb_train, zero_dim_ttb_test, num_direc=p, bandwidth=s)
        X_train_features_0_btt_swk, X_test_features_0_btt_swk = swk_features(zero_dim_btt_train, zero_dim_btt_test, num_direc=p, bandwidth=s)

        X_train_features_1_ltr_swk, X_test_features_1_ltr_swk = swk_features(one_dim_ltr_train, one_dim_ltr_test, num_direc=p, bandwidth=s)
        X_train_features_1_rtl_swk, X_test_features_1_rtl_swk = swk_features(one_dim_rtl_train, one_dim_rtl_test, num_direc=p, bandwidth=s)
        X_train_features_1_ttb_swk, X_test_features_1_ttb_swk = swk_features(one_dim_ttb_train, one_dim_ttb_test, num_direc=p, bandwidth=s)
        X_train_features_1_btt_swk, X_test_features_1_btt_swk = swk_features(one_dim_btt_train, one_dim_btt_test, num_direc=p, bandwidth=s)
        
        X_train_features = np.column_stack((X_train_features_1_ltr_swk,X_train_features_1_rtl_swk,X_train_features_1_ttb_swk,X_train_features_1_btt_swk,X_train_features_0_ltr_swk,X_train_features_0_rtl_swk,X_train_features_0_btt_swk,X_train_features_0_ttb_swk))
        X_test_features = np.column_stack((X_test_features_1_ltr_swk,X_test_features_1_rtl_swk,X_test_features_1_ttb_swk,X_test_features_1_btt_swk,X_test_features_0_ltr_swk,X_test_features_0_rtl_swk,X_test_features_0_btt_swk,X_test_features_0_ttb_swk))

        c = [20]
        for i in c:
            clf = SVC(kernel='rbf', C=i).fit(X_train_features, y_train)
            train_accuracy.append(clf.score(X_train_features, y_train))
            test_accuracy.append(clf.score(X_test_features, y_test))
            p_model.append(p)
            s_model.append(s)
            model_type.append('SVC')
            c_model.append(i)
            
swk_results = pd.DataFrame()
swk_results['Training Accuracy'] = train_accuracy
swk_results['Test Accuracy'] = test_accuracy
swk_results['p'] = p_model
swk_results['s'] = s_model
swk_results['c'] = c_model
swk_results['Model Type'] = model_type

swk_sorted = swk_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(swk_sorted[0:60])

"""
##========================================================##
##SILHOUETTE METHOD

"""
resolution = [500]

train_accuracy = []
test_accuracy = []
r_model = []
c_model = []
model_type = []

for j in resolution:
    X_train_features_1_ltr_silhouettes, X_test_features_1_ltr_silhouettes = silhouette_features(one_dim_ltr_train, one_dim_ltr_test, resolution=j)
    X_train_features_0_ltr_silhouettes, X_test_features_0_ltr_silhouettes = silhouette_features(zero_dim_ltr_train, zero_dim_ltr_test, resolution=j)
    X_train_features_1_rtl_silhouettes, X_test_features_1_rtl_silhouettes = silhouette_features(one_dim_rtl_train, one_dim_rtl_test, resolution=j)
    X_train_features_0_rtl_silhouettes, X_test_features_0_rtl_silhouettes = silhouette_features(zero_dim_rtl_train, zero_dim_rtl_test, resolution=j)

    X_train_features_1_ttb_silhouettes, X_test_features_1_ttb_silhouettes = silhouette_features(one_dim_ttb_train, one_dim_ttb_test, resolution=j)
    X_train_features_0_ttb_silhouettes, X_test_features_0_ttb_silhouettes = silhouette_features(zero_dim_ttb_train, zero_dim_ttb_test, resolution=j)

    X_train_features_1_btt_silhouettes, X_test_features_1_btt_silhouettes = silhouette_features(one_dim_btt_train, one_dim_btt_test, resolution=j)
    X_train_features_0_btt_silhouettes, X_test_features_0_btt_silhouettes = silhouette_features(zero_dim_btt_train, zero_dim_btt_test, resolution=j)
        
    X_train_features = np.column_stack((X_train_features_1_ltr_silhouettes,X_train_features_1_rtl_silhouettes,X_train_features_1_ttb_silhouettes,X_train_features_1_btt_silhouettes,X_train_features_0_ltr_silhouettes,X_train_features_0_rtl_silhouettes,X_train_features_0_btt_silhouettes,X_train_features_0_ttb_silhouettes))
    X_test_features = np.column_stack((X_test_features_1_ltr_silhouettes,X_test_features_1_rtl_silhouettes,X_test_features_1_ttb_silhouettes,X_test_features_1_btt_silhouettes,X_test_features_0_ltr_silhouettes,X_test_features_0_rtl_silhouettes,X_test_features_0_btt_silhouettes,X_test_features_0_ttb_silhouettes))
    c = [12]
    for k in c:
       clf = SVC(kernel='rbf', C=k).fit(X_train_features, y_train)
       train_accuracy.append(clf.score(X_train_features, y_train))
       test_accuracy.append(clf.score(X_test_features, y_test))
       r_model.append(j)
       model_type.append('SVC')
       c_model.append(k)

silhouette_results = pd.DataFrame()
silhouette_results['Training Accuracy'] = train_accuracy
silhouette_results['Test Accuracy'] = test_accuracy
silhouette_results['r'] = r_model
silhouette_results['c'] = c_model
silhouette_results['Model Type'] = model_type

silhouette_sorted = silhouette_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(silhouette_sorted[0:50])

"""

##=======================================================##
##PERSISTENCE WEIGHTED GAUSSIAN KERNEL METHOD

"""

bandwidth = [5]

train_accuracy = []
test_accuracy = []
p_model = []
s_model = []
c_model = []
model_type = []

for s in bandwidth:
    X_train_features_0_ltr_pwgk, X_test_features_0_ltr_pwgk = pwgk_features(zero_dim_ltr_train, zero_dim_ltr_test, bandwidth=s)
    X_train_features_0_rtl_pwgk, X_test_features_0_rtl_pwgk = pwgk_features(zero_dim_rtl_train, zero_dim_rtl_test, bandwidth=s)
    X_train_features_0_ttb_pwgk, X_test_features_0_ttb_pwgk = pwgk_features(zero_dim_ttb_train, zero_dim_ttb_test, bandwidth=s)
    X_train_features_0_btt_pwgk, X_test_features_0_btt_pwgk = pwgk_features(zero_dim_btt_train, zero_dim_btt_test, bandwidth=s)
    X_train_features_1_ltr_pwgk, X_test_features_1_ltr_pwgk = pwgk_features(one_dim_ltr_train, one_dim_ltr_test, bandwidth=s)
    X_train_features_1_rtl_pwgk, X_test_features_1_rtl_pwgk = pwgk_features(one_dim_rtl_train, one_dim_rtl_test, bandwidth=s)
    X_train_features_1_ttb_pwgk, X_test_features_1_ttb_pwgk = pwgk_features(one_dim_ttb_train, one_dim_ttb_test, bandwidth=s)
    X_train_features_1_btt_pwgk, X_test_features_1_btt_pwgk = pwgk_features(one_dim_btt_train, one_dim_btt_test, bandwidth=s)
        
    X_train_features = np.column_stack((X_train_features_1_ltr_pwgk,X_train_features_1_rtl_pwgk,X_train_features_1_ttb_pwgk,X_train_features_1_btt_pwgk,X_train_features_0_ltr_pwgk,X_train_features_0_rtl_pwgk,X_train_features_0_btt_pwgk,X_train_features_0_ttb_pwgk))
    X_test_features = np.column_stack((X_test_features_1_ltr_pwgk,X_test_features_1_rtl_pwgk,X_test_features_1_ttb_pwgk,X_test_features_1_btt_pwgk,X_test_features_0_ltr_pwgk,X_test_features_0_rtl_pwgk,X_test_features_0_btt_pwgk,X_test_features_0_ttb_pwgk))

    c = [30]
    for i in c:
        clf = SVC(kernel='rbf', C=i).fit(X_train_features, y_train)
        train_accuracy.append(clf.score(X_train_features, y_train))
        test_accuracy.append(clf.score(X_test_features, y_test))
        s_model.append(s)
        model_type.append('SVC')
        c_model.append(i)
            
pwgk_results = pd.DataFrame()
pwgk_results['Training Accuracy'] = train_accuracy
pwgk_results['Test Accuracy'] = test_accuracy
pwgk_results['s'] = s_model
pwgk_results['c'] = c_model
pwgk_results['Model Type'] = model_type

pwgk_sorted = pwgk_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(pwgk_sorted[0:60])
"""

##========================================================##
##PERSISTENCE SCALE SPACE KERNEL METHOD

"""

bandwidth = [20]

train_accuracy = []
test_accuracy = []
p_model = []
s_model = []
c_model = []
model_type = []

for s in bandwidth:
    X_train_features_0_ltr_pssk, X_test_features_0_ltr_pssk = pssk_features(zero_dim_ltr_train, zero_dim_ltr_test, bandwidth=s)
    X_train_features_0_rtl_pssk, X_test_features_0_rtl_pssk = pssk_features(zero_dim_rtl_train, zero_dim_rtl_test, bandwidth=s)
    X_train_features_0_ttb_pssk, X_test_features_0_ttb_pssk = pssk_features(zero_dim_ttb_train, zero_dim_ttb_test, bandwidth=s)
    X_train_features_0_btt_pssk, X_test_features_0_btt_pssk = pssk_features(zero_dim_btt_train, zero_dim_btt_test, bandwidth=s)
    X_train_features_1_ltr_pssk, X_test_features_1_ltr_pssk = pssk_features(one_dim_ltr_train, one_dim_ltr_test, bandwidth=s)
    X_train_features_1_rtl_pssk, X_test_features_1_rtl_pssk = pssk_features(one_dim_rtl_train, one_dim_rtl_test, bandwidth=s)
    X_train_features_1_ttb_pssk, X_test_features_1_ttb_pssk = pssk_features(one_dim_ttb_train, one_dim_ttb_test, bandwidth=s)
    X_train_features_1_btt_pssk, X_test_features_1_btt_pssk = pssk_features(one_dim_btt_train, one_dim_btt_test, bandwidth=s)
        
    X_train_features = np.column_stack((X_train_features_1_ltr_pssk,X_train_features_1_rtl_pssk,X_train_features_1_ttb_pssk,X_train_features_1_btt_pssk,X_train_features_0_ltr_pssk,X_train_features_0_rtl_pssk,X_train_features_0_btt_pssk,X_train_features_0_ttb_pssk))
    X_test_features = np.column_stack((X_test_features_1_ltr_pssk,X_test_features_1_rtl_pssk,X_test_features_1_ttb_pssk,X_test_features_1_btt_pssk,X_test_features_0_ltr_pssk,X_test_features_0_rtl_pssk,X_test_features_0_btt_pssk,X_test_features_0_ttb_pssk))

    c = [40]
    for i in c:
        clf = SVC(kernel='rbf', C=i).fit(X_train_features, y_train)
        train_accuracy.append(clf.score(X_train_features, y_train))
        test_accuracy.append(clf.score(X_test_features, y_test))
        s_model.append(s)
        model_type.append('SVC')
        c_model.append(i)
            
pssk_results = pd.DataFrame()
pssk_results['Training Accuracy'] = train_accuracy
pssk_results['Test Accuracy'] = test_accuracy
pssk_results['s'] = s_model
pssk_results['c'] = c_model
pssk_results['Model Type'] = model_type

pssk_sorted = pssk_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(pssk_sorted[0:60])

"""
##=========================================================##
##PERSISTENCE FISHER KERNEL METHOD

"""

band_fisher = [1]
bandwidth = [20]

train_accuracy = []
test_accuracy = []
bf_model = []
b_model = []
c_model = []
model_type = []

for j in band_fisher:
    for s in bandwidth:
        X_train_features_0_ltr_pfk, X_test_features_0_ltr_pfk = pfk_features(zero_dim_ltr_train, zero_dim_ltr_test, bandwidth=s, bandwidth_fisher=j)
        X_train_features_0_rtl_pfk, X_test_features_0_rtl_pfk = pfk_features(zero_dim_rtl_train, zero_dim_rtl_test, bandwidth=s, bandwidth_fisher=j)
        X_train_features_0_ttb_pfk, X_test_features_0_ttb_pfk = pfk_features(zero_dim_ttb_train, zero_dim_ttb_test, bandwidth=s, bandwidth_fisher=j)
        X_train_features_0_btt_pfk, X_test_features_0_btt_pfk = pfk_features(zero_dim_btt_train, zero_dim_btt_test, bandwidth=s, bandwidth_fisher=j)
        X_train_features_1_ltr_pfk, X_test_features_1_ltr_pfk = pfk_features(one_dim_ltr_train, one_dim_ltr_test, bandwidth=s, bandwidth_fisher=j)
        X_train_features_1_rtl_pfk, X_test_features_1_rtl_pfk = pfk_features(one_dim_rtl_train, one_dim_rtl_test, bandwidth=s, bandwidth_fisher=j)
        X_train_features_1_ttb_pfk, X_test_features_1_ttb_pfk = pfk_features(one_dim_ttb_train, one_dim_ttb_test, bandwidth=s, bandwidth_fisher=j)
        X_train_features_1_btt_pfk, X_test_features_1_btt_pfk = pfk_features(one_dim_btt_train, one_dim_btt_test, bandwidth=s, bandwidth_fisher=j)
            
        X_train_features = np.column_stack((X_train_features_1_ltr_pfk,X_train_features_1_rtl_pfk,X_train_features_1_ttb_pfk,X_train_features_1_btt_pfk,X_train_features_0_ltr_pfk,X_train_features_0_rtl_pfk,X_train_features_0_btt_pfk,X_train_features_0_ttb_pfk))
        X_test_features = np.column_stack((X_test_features_1_ltr_pfk,X_test_features_1_rtl_pfk,X_test_features_1_ttb_pfk,X_test_features_1_btt_pfk,X_test_features_0_ltr_pfk,X_test_features_0_rtl_pfk,X_test_features_0_btt_pfk,X_test_features_0_ttb_pfk))
    
        c = [40]
        for i in c:
            clf = SVC(kernel='rbf', C=i).fit(X_train_features, y_train)
            train_accuracy.append(clf.score(X_train_features, y_train))
            test_accuracy.append(clf.score(X_test_features, y_test))
            bf_model.append(j)
            b_model.append(s)
            model_type.append('SVC')
            c_model.append(i)
            
pfk_results = pd.DataFrame()
pfk_results['Training Accuracy'] = train_accuracy
pfk_results['Test Accuracy'] = test_accuracy
pfk_results['bf'] = bf_model
pfk_results['b'] = b_model
pfk_results['c'] = c_model
pfk_results['Model Type'] = model_type

pfk_sorted = pfk_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(pfk_sorted[0:60])

"""

##========================================================##
##PERSISTENT ENTROPY METHOD


resolution = [200]

train_accuracy = []
test_accuracy = []
r_model = []
c_model = []
model_type = []

for j in resolution:
    X_train_features_1_ltr_entropies, X_test_features_1_ltr_entropies = entropy_features(one_dim_ltr_train, one_dim_ltr_test, resolution=j)
    X_train_features_0_ltr_entropies, X_test_features_0_ltr_entropies = entropy_features(zero_dim_ltr_train, zero_dim_ltr_test, resolution=j)
    X_train_features_1_rtl_entropies, X_test_features_1_rtl_entropies = entropy_features(one_dim_rtl_train, one_dim_rtl_test, resolution=j)
    X_train_features_0_rtl_entropies, X_test_features_0_rtl_entropies = entropy_features(zero_dim_rtl_train, zero_dim_rtl_test, resolution=j)

    X_train_features_1_ttb_entropies, X_test_features_1_ttb_entropies = entropy_features(one_dim_ttb_train, one_dim_ttb_test, resolution=j)
    X_train_features_0_ttb_entropies, X_test_features_0_ttb_entropies = entropy_features(zero_dim_ttb_train, zero_dim_ttb_test, resolution=j)

    X_train_features_1_btt_entropies, X_test_features_1_btt_entropies = entropy_features(one_dim_btt_train, one_dim_btt_test, resolution=j)
    X_train_features_0_btt_entropies, X_test_features_0_btt_entropies = entropy_features(zero_dim_btt_train, zero_dim_btt_test, resolution=j)
        
    X_train_features = np.column_stack((X_train_features_1_ltr_entropies,X_train_features_1_rtl_entropies,X_train_features_1_ttb_entropies,X_train_features_1_btt_entropies,X_train_features_0_ltr_entropies,X_train_features_0_rtl_entropies,X_train_features_0_btt_entropies,X_train_features_0_ttb_entropies))
    X_test_features = np.column_stack((X_test_features_1_ltr_entropies,X_test_features_1_rtl_entropies,X_test_features_1_ttb_entropies,X_test_features_1_btt_entropies,X_test_features_0_ltr_entropies,X_test_features_0_rtl_entropies,X_test_features_0_btt_entropies,X_test_features_0_ttb_entropies))
    c = [40]
    for k in c:
       clf = SVC(kernel='rbf', C=k).fit(X_train_features, y_train)
       train_accuracy.append(clf.score(X_train_features, y_train))
       test_accuracy.append(clf.score(X_test_features, y_test))
       r_model.append(j)
       model_type.append('SVC')
       c_model.append(k)

entropy_results = pd.DataFrame()
entropy_results['Training Accuracy'] = train_accuracy
entropy_results['Test Accuracy'] = test_accuracy
entropy_results['r'] = r_model
entropy_results['c'] = c_model
entropy_results['Model Type'] = model_type

entropy_sorted = entropy_results.sort_values(by=['Test Accuracy', 'Training Accuracy'],ascending=False)
print(entropy_sorted[0:50])


##=====================================================##
##CARLSSON COORDINATES

"""

X_train_features_0_ltr_cc1, X_train_features_0_ltr_cc2, X_train_features_0_ltr_cc3, X_train_features_0_ltr_cc4, X_test_features_0_ltr_cc1, X_test_features_0_ltr_cc2, X_test_features_0_ltr_cc3, X_test_features_0_ltr_cc4 = carlsson_coordinates(zero_dim_ltr_train, zero_dim_ltr_test)
X_train_features_1_ltr_cc1, X_train_features_1_ltr_cc2, X_train_features_1_ltr_cc3, X_train_features_1_ltr_cc4, X_test_features_1_ltr_cc1, X_test_features_1_ltr_cc2, X_test_features_1_ltr_cc3, X_test_features_1_ltr_cc4 = carlsson_coordinates(one_dim_ltr_train, one_dim_ltr_test)
X_train_features_0_rtl_cc1, X_train_features_0_rtl_cc2, X_train_features_0_rtl_cc3, X_train_features_0_rtl_cc4, X_test_features_0_rtl_cc1, X_test_features_0_rtl_cc2, X_test_features_0_rtl_cc3, X_test_features_0_rtl_cc4 = carlsson_coordinates(zero_dim_rtl_train, zero_dim_rtl_test)
X_train_features_1_rtl_cc1, X_train_features_1_rtl_cc2, X_train_features_1_rtl_cc3, X_train_features_1_rtl_cc4, X_test_features_1_rtl_cc1, X_test_features_1_rtl_cc2, X_test_features_1_rtl_cc3, X_test_features_1_rtl_cc4 = carlsson_coordinates(one_dim_rtl_train, one_dim_rtl_test)
X_train_features_0_btt_cc1, X_train_features_0_btt_cc2, X_train_features_0_btt_cc3, X_train_features_0_btt_cc4, X_test_features_0_btt_cc1, X_test_features_0_btt_cc2, X_test_features_0_btt_cc3, X_test_features_0_btt_cc4 = carlsson_coordinates(zero_dim_btt_train, zero_dim_btt_test)
X_train_features_1_btt_cc1, X_train_features_1_btt_cc2, X_train_features_1_btt_cc3, X_train_features_1_btt_cc4, X_test_features_1_btt_cc1, X_test_features_1_btt_cc2, X_test_features_1_btt_cc3, X_test_features_1_btt_cc4 = carlsson_coordinates(one_dim_btt_train, one_dim_btt_test)
X_train_features_0_ttb_cc1, X_train_features_0_ttb_cc2, X_train_features_0_ttb_cc3, X_train_features_0_ttb_cc4, X_test_features_0_ttb_cc1, X_test_features_0_ttb_cc2, X_test_features_0_ttb_cc3, X_test_features_0_ttb_cc4 = carlsson_coordinates(zero_dim_ttb_train, zero_dim_ttb_test)
X_train_features_1_ttb_cc1, X_train_features_1_ttb_cc2, X_train_features_1_ttb_cc3, X_train_features_1_ttb_cc4, X_test_features_1_ttb_cc1, X_test_features_1_ttb_cc2, X_test_features_1_ttb_cc3, X_test_features_1_ttb_cc4 = carlsson_coordinates(one_dim_ttb_train, one_dim_ttb_test)

X_train_features = np.column_stack((scale(X_train_features_0_ltr_cc1), scale(X_train_features_0_ltr_cc2),scale(X_train_features_0_ltr_cc3),scale(X_train_features_0_ltr_cc4),
                                   scale(X_train_features_0_rtl_cc1), scale(X_train_features_0_rtl_cc2),scale(X_train_features_0_rtl_cc3),scale(X_train_features_0_rtl_cc4),
                                   scale(X_train_features_0_ttb_cc1), scale(X_train_features_0_ttb_cc2),scale(X_train_features_0_ttb_cc3),scale(X_train_features_0_ttb_cc4),
                                   scale(X_train_features_0_btt_cc1), scale(X_train_features_0_btt_cc2),scale(X_train_features_0_btt_cc3),scale(X_train_features_0_btt_cc4),
                                   scale(X_train_features_1_ltr_cc1), scale(X_train_features_1_ltr_cc2),scale(X_train_features_1_ltr_cc3),scale(X_train_features_1_ltr_cc4),
                                   scale(X_train_features_1_rtl_cc1), scale(X_train_features_1_rtl_cc2),scale(X_train_features_1_rtl_cc3),scale(X_train_features_1_rtl_cc4),
                                   scale(X_train_features_1_ttb_cc1), scale(X_train_features_1_ttb_cc2),scale(X_train_features_1_ttb_cc3),scale(X_train_features_1_ttb_cc4),
                                   scale(X_train_features_1_btt_cc1), scale(X_train_features_1_btt_cc2),scale(X_train_features_1_btt_cc3),scale(X_train_features_1_btt_cc4)))

X_test_features = np.column_stack((scale(X_test_features_0_ltr_cc1), scale(X_test_features_0_ltr_cc2), scale(X_test_features_0_ltr_cc3), scale(X_test_features_0_ltr_cc4),
                                  scale(X_test_features_0_rtl_cc1), scale(X_test_features_0_rtl_cc2), scale(X_test_features_0_rtl_cc3), scale(X_test_features_0_rtl_cc4),
                                   scale(X_test_features_0_ttb_cc1), scale(X_test_features_0_ttb_cc2), scale(X_test_features_0_ttb_cc3), scale(X_test_features_0_ttb_cc4),
                                  scale(X_test_features_0_btt_cc1), scale(X_test_features_0_btt_cc2), scale(X_test_features_0_btt_cc3), scale(X_test_features_0_btt_cc4),
                                  scale(X_test_features_1_ltr_cc1), scale(X_test_features_1_ltr_cc2), scale(X_test_features_1_ltr_cc3), scale(X_test_features_1_ltr_cc4),
                                  scale(X_test_features_1_rtl_cc1), scale(X_test_features_1_rtl_cc2), scale(X_test_features_1_rtl_cc3), scale(X_test_features_1_rtl_cc4),
                                   scale(X_test_features_1_ttb_cc1), scale(X_test_features_1_ttb_cc2), scale(X_test_features_1_ttb_cc3), scale(X_test_features_1_ttb_cc4),
                                  scale(X_test_features_1_btt_cc1), scale(X_test_features_1_btt_cc2), scale(X_test_features_1_btt_cc3), scale(X_test_features_1_btt_cc4)))

clf = SVC(C=50).fit(X_train_features, y_train)

print('Train/test classification accuracy with persistence functions')
print(clf.score(X_train_features, y_train))
print(clf.score(X_test_features, y_test))

"""

##========================================================##
##TROPICAL COORDINATES

"""

X_train_0_ltr_tc1, X_train_0_ltr_tc3, X_train_0_ltr_tc4, X_train_0_ltr_tc5, X_train_0_ltr_tc7, X_test_0_ltr_tc1, X_test_0_ltr_tc3, X_test_0_ltr_tc4, X_test_0_ltr_tc5, X_test_0_ltr_tc7 = tropical_coordinates(zero_dim_ltr_train, zero_dim_ltr_test)
X_train_1_ltr_tc1, X_train_1_ltr_tc3, X_train_1_ltr_tc4, X_train_1_ltr_tc5, X_train_1_ltr_tc7, X_test_1_ltr_tc1,X_test_1_ltr_tc3, X_test_1_ltr_tc4, X_test_1_ltr_tc5,  X_test_1_ltr_tc7 = tropical_coordinates(one_dim_ltr_train, one_dim_ltr_test)
X_train_0_rtl_tc1, X_train_0_rtl_tc3, X_train_0_rtl_tc4, X_train_0_rtl_tc5, X_train_0_rtl_tc7, X_test_0_rtl_tc1,  X_test_0_rtl_tc3, X_test_0_rtl_tc4, X_test_0_rtl_tc5, X_test_0_rtl_tc7 = tropical_coordinates(zero_dim_rtl_train, zero_dim_rtl_test)
X_train_1_rtl_tc1, X_train_1_rtl_tc3, X_train_1_rtl_tc4, X_train_1_rtl_tc5, X_train_1_rtl_tc7, X_test_1_rtl_tc1,  X_test_1_rtl_tc3, X_test_1_rtl_tc4, X_test_1_rtl_tc5, X_test_1_rtl_tc7 = tropical_coordinates(one_dim_rtl_train, one_dim_rtl_test)
X_train_0_btt_tc1, X_train_0_btt_tc3, X_train_0_btt_tc4, X_train_0_btt_tc5, X_train_0_btt_tc7, X_test_0_btt_tc1, X_test_0_btt_tc3, X_test_0_btt_tc4, X_test_0_btt_tc5,  X_test_0_btt_tc7 = tropical_coordinates(zero_dim_btt_train, zero_dim_btt_test)
X_train_1_btt_tc1, X_train_1_btt_tc3, X_train_1_btt_tc4, X_train_1_btt_tc5, X_train_1_btt_tc7, X_test_1_btt_tc1,  X_test_1_btt_tc3, X_test_1_btt_tc4, X_test_1_btt_tc5,  X_test_1_btt_tc7 = tropical_coordinates(one_dim_btt_train, one_dim_btt_test)
X_train_0_ttb_tc1, X_train_0_ttb_tc3, X_train_0_ttb_tc4, X_train_0_ttb_tc5, X_train_0_ttb_tc7, X_test_0_ttb_tc1, X_test_0_ttb_tc3, X_test_0_ttb_tc4, X_test_0_ttb_tc5,  X_test_0_ttb_tc7 = tropical_coordinates(zero_dim_ttb_train, zero_dim_ttb_test)
X_train_1_ttb_tc1, X_train_1_ttb_tc3, X_train_1_ttb_tc4, X_train_1_ttb_tc5, X_train_1_ttb_tc7, X_test_1_ttb_tc1, X_test_1_ttb_tc3, X_test_1_ttb_tc4, X_test_1_ttb_tc5,  X_test_1_ttb_tc7 = tropical_coordinates(one_dim_ttb_train, one_dim_ttb_test)

X_train_features = np.column_stack((scale(X_train_0_ltr_tc1),scale(X_train_0_ltr_tc3),scale(X_train_0_ltr_tc4), scale(X_train_0_ltr_tc5), scale(X_train_0_ltr_tc7),
                                   scale(X_train_0_rtl_tc1), scale(X_train_0_rtl_tc3),scale(X_train_0_rtl_tc4), scale(X_train_0_rtl_tc5), scale(X_train_0_rtl_tc7),
                                   scale(X_train_0_ttb_tc1), scale(X_train_0_ttb_tc3),scale(X_train_0_ttb_tc4), scale(X_train_0_ttb_tc5), scale(X_train_0_ttb_tc7),
                                   scale(X_train_0_btt_tc1), scale(X_train_0_btt_tc3),scale(X_train_0_btt_tc4), scale(X_train_0_btt_tc5), scale(X_train_0_btt_tc7),
                                   scale(X_train_1_ltr_tc1), scale(X_train_1_ltr_tc3),scale(X_train_1_ltr_tc4), scale(X_train_1_ltr_tc5), scale(X_train_1_ltr_tc7),
                                   scale(X_train_1_rtl_tc1), scale(X_train_1_rtl_tc3),scale(X_train_1_rtl_tc4), scale(X_train_1_rtl_tc5), scale(X_train_1_rtl_tc7),
                                   scale(X_train_1_ttb_tc1), scale(X_train_1_ttb_tc3),scale(X_train_1_ttb_tc4), scale(X_train_1_ttb_tc5), scale(X_train_1_ttb_tc7),
                                   scale(X_train_1_btt_tc1), scale(X_train_1_btt_tc3),scale(X_train_1_btt_tc4), scale(X_train_1_btt_tc5), scale(X_train_1_btt_tc7)))

X_test_features = np.column_stack((scale(X_test_0_ltr_tc1), scale(X_test_0_ltr_tc3), scale(X_test_0_ltr_tc4), scale(X_test_0_ltr_tc5), scale(X_test_0_ltr_tc7),
                                  scale(X_test_0_rtl_tc1),  scale(X_test_0_rtl_tc3), scale(X_test_0_rtl_tc4), scale(X_test_0_rtl_tc5),  scale(X_test_0_rtl_tc7),
                                   scale(X_test_0_ttb_tc1), scale(X_test_0_ttb_tc3), scale(X_test_0_ttb_tc4), scale(X_test_0_ttb_tc5),  scale(X_test_0_ttb_tc7),
                                  scale(X_test_0_btt_tc1), scale(X_test_0_btt_tc3), scale(X_test_0_btt_tc4), scale(X_test_0_btt_tc5),  scale(X_test_0_btt_tc7),
                                  scale(X_test_1_ltr_tc1), scale(X_test_1_ltr_tc3), scale(X_test_1_ltr_tc4), scale(X_test_1_ltr_tc5),  scale(X_test_1_ltr_tc7),
                                  scale(X_test_1_rtl_tc1), scale(X_test_1_rtl_tc3), scale(X_test_1_rtl_tc4), scale(X_test_1_rtl_tc5), scale(X_test_1_rtl_tc7),
                                   scale(X_test_1_ttb_tc1), scale(X_test_1_ttb_tc3), scale(X_test_1_ttb_tc4), scale(X_test_1_ttb_tc5),  scale(X_test_1_ttb_tc7),
                                  scale(X_test_1_btt_tc1), scale(X_test_1_btt_tc3), scale(X_test_1_btt_tc4), scale(X_test_1_btt_tc5), scale(X_test_1_btt_tc7)))

clf = SVC(C=10).fit(X_train_features, y_train)

print('Train/test classification accuracy with persistence functions')
print(clf.score(X_train_features, y_train))
print(clf.score(X_test_features, y_test))

"""

##========================================================##

##List of parameters used for the results obtained in the master thesis
"""

'BEST' PARAMETERS:
    SWK: - num_directions = 10
       - bandwidth = 10000
       - SVC constant = 20
    
    PWGK: - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 5
       - SVC constant = 30 
    
    PSSK: - bandwidth = 20
       - SVC constant = 40
    
    PFK : - bandwidth_fisher = default
       - bandwidth = 20
       - SVC constant = 40 
       
    Landscape: - num_landscapes = 20
       - resolution = 500
       - SVC constant = 80
       
    Persistence Images: - resolution = [30,30]
       - bandwidth = 10
       - weight = default
       - SVC constant = 30
       
    Persistence Silhouette: - resolution = 500
       - weight = default
       - SVC constant = 12
    
    Carlsson Coordinates: SVC = 50
    
    Tropical Coordinates: SVC = 10
    
    Persistent Entropy: - resolution = 200
       - mode = vector
       - SVC constant =  40
       - normalized = False

"""

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")