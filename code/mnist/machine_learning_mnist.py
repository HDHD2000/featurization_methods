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

from persistence_methods_numba import reshape_persistence_diagrams
from persistence_methods import kernel_features
from persistence_methods import tent_features
from persistence_methods import carlsson_coordinates
from persistence_methods import landscape_features
from persistence_methods import persistence_image_features
from persistence_methods import fast_kernel_features
from persistence_methods_numba import numba_kernel_features_train
from persistence_methods_numba import numba_kernel_features_test

start = time.time()

(train_X, train_y), (test_X, test_y) = mnist.load_data()


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

def compute_persistence(directional_transform):
    simplex_tree = gd.RipsComplex(directional_transform).create_simplex_tree(max_dimension = 2)
    barcode = simplex_tree.persistence()
    dgms_lower_0 = simplex_tree.persistence_intervals_in_dimension(0)
    dgms_lower_1 = simplex_tree.persistence_intervals_in_dimension(1)
    if len(dgms_lower_1) == 0:
        dgms_lower_1 = np.array([0.0,0.0]).reshape((1,2))
    dgms_lower = [dgms_lower_0, dgms_lower_1]
    return dgms_lower

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
    
#print(np.shape(np.array([0.0,0.0]).reshape((1,2))))

n=110
label = test_y[n]
print("label: ", label)
imgs = directional_transform(test_X[n])
f, axarr = plt.subplots(1,4)
ltr = axarr[0].imshow(imgs[0])
ltr.axes.get_xaxis().set_visible(False)
ltr.axes.get_yaxis().set_visible(False)
rtl = axarr[1].imshow(imgs[1])
rtl.axes.get_xaxis().set_visible(False)
rtl.axes.get_yaxis().set_visible(False)
ttb = axarr[2].imshow(imgs[2])
ttb.axes.get_xaxis().set_visible(False)
ttb.axes.get_yaxis().set_visible(False)
btt = axarr[3].imshow(imgs[3])
btt.axes.get_xaxis().set_visible(False)
btt.axes.get_yaxis().set_visible(False)
plt.savefig('mnist_eight_sweeps.png')

tuning = np.arange(0,1000)
unused_index, tuning_index = train_test_split(tuning, test_size = .1, random_state=1)
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
            
train_index, test_index, original_index_train, original_index_test, y_train, y_test = train_test_split(obs, tuning_index, test_y[tuning_index], test_size = .2, random_state=1, stratify = train_y[tuning_index])

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


""" 
n = [10]
r = [50,100]

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

        c = [20]
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
  
pixels = [[30,30]]
spread = [1]

train_accuracy = []
test_accuracy = []
p_model = []
s_model = []
c_model = []
model_type = []

for p in pixels:
    for s in spread:
        X_train_features_0_ltr_imgs, X_test_features_0_ltr_imgs = persistence_image_features(zero_dim_ltr_train, zero_dim_ltr_test, pixels=p, spread=s)
        X_train_features_0_rtl_imgs, X_test_features_0_rtl_imgs = persistence_image_features(zero_dim_rtl_train, zero_dim_rtl_test, pixels=p, spread=s)
        X_train_features_0_ttb_imgs, X_test_features_0_ttb_imgs = persistence_image_features(zero_dim_ttb_train, zero_dim_ttb_test, pixels=p, spread=s)
        X_train_features_0_btt_imgs, X_test_features_0_btt_imgs = persistence_image_features(zero_dim_btt_train, zero_dim_btt_test, pixels=p, spread=s)

        X_train_features_1_ltr_imgs, X_test_features_1_ltr_imgs = persistence_image_features(one_dim_ltr_train, one_dim_ltr_test, pixels=p, spread=s)
        X_train_features_1_rtl_imgs, X_test_features_1_rtl_imgs = persistence_image_features(one_dim_rtl_train, one_dim_rtl_test, pixels=p, spread=s)
        X_train_features_1_ttb_imgs, X_test_features_1_ttb_imgs = persistence_image_features(one_dim_ttb_train, one_dim_ttb_test, pixels=p, spread=s)
        X_train_features_1_btt_imgs, X_test_features_1_btt_imgs = persistence_image_features(one_dim_btt_train, one_dim_btt_test, pixels=p, spread=s)
        
        X_train_features = np.column_stack((X_train_features_1_ltr_imgs,X_train_features_1_rtl_imgs,X_train_features_1_ttb_imgs,X_train_features_1_btt_imgs,X_train_features_0_ltr_imgs,X_train_features_0_rtl_imgs,X_train_features_0_btt_imgs,X_train_features_0_ttb_imgs))
        X_test_features = np.column_stack((X_test_features_1_ltr_imgs,X_test_features_1_rtl_imgs,X_test_features_1_ttb_imgs,X_test_features_1_btt_imgs,X_test_features_0_ltr_imgs,X_test_features_0_rtl_imgs,X_test_features_0_btt_imgs,X_test_features_0_ttb_imgs))

        c = [20]
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

pass 
end = time.time()
delta = end - start
print("took " + str(delta) + " seconds to process")


"""

'BEST' PARAMETERS:
    SWK: - num_directions = 
       - bandwidth = 
       - SVC constant = 
    
    PWGK: - weight = lambda x: np.arctan(x[1]-x[0])
       - bandwidth = 
       - SVC constant = 
    
    PSSK: - bandwidth = 
       - SVC constant = 
    
    PFK : - bandwidth_fisher = 
       - bandwidth = 
       - SVC constant = 
       
    Landscape: - num_landscapes = 10
       - resolution = 50
       - SVC constant = 20
       
    Persistence Images: - resolution = 
       - bandwidth = 
       - weight = lambda x: x[1]**2
       - SVC constant = 
       
    Persistence Silhouette: - resolution = 
       - weight = 
       - SVC constant = 
    
    Persistent Entropy: - resolution = 
       - mode = 
       - SVC constant = 

"""