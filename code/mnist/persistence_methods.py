from persim import heat
import numpy as np
import ATS
import time
from sklearn import mixture
from gudhi.representations.vector_methods import Landscape
from gudhi.representations.vector_methods import PersistenceImage
from persim import PersImage
from sklearn.neighbors import DistanceMetric

def kernel_features(train, test, s):
    n_train = len(train)
    n_test = len(test)
    X_train_features = np.zeros((n_train, n_train))
    X_test_features = np.zeros((n_test, n_train))
    
    start = time.time()
    for i in range(0,n_train):
        for j in range(0,i):
            print("train: ", j)
            dgm0 = train[i]
            dgm1 = train[j]
            hk = heat(dgm0, dgm1, sigma = s) 
            X_train_features[i,j] = hk
            X_train_features[j,i] = hk
        for j in range(0,n_test):
            print("test: ", j)
            dgm0 = train[i]
            dgm1 = test[j]
            hk = heat(dgm0, dgm1, sigma = s)        
            X_test_features[j,i] = hk

    print(time.time()-start)
    return X_train_features, X_test_features

def tent_features(X_train, X_test, d=5, padding=.05):
    centers, delta = ATS.box_centers(X_train, d, padding) 

    start = time.time()

    X_train_features = ATS.get_all_features_boxes(X_train, centers, delta)

    X_test_features = ATS.get_all_features_boxes(X_test, centers, delta)

    end = time.time()
    print('Computing features took (seconds):{}'.format(end-start))
    return X_train_features, X_test_features

def carlsson_coordinates(X_train, X_test):
    n = len(X_train)
    X_train_features_cc1 = np.zeros(n)
    X_train_features_cc2 = np.zeros(n)
    X_train_features_cc3 = np.zeros(n)
    X_train_features_cc4 = np.zeros(n)
    start = time.time()
    ymax = 0
    for i in range(0,n):
        if len(X_train[i])>0:
            b = np.max(X_train[i][:,1])
        else:
            b = ymax
        if ymax < b:
            ymax = b
        else:
            ymax = ymax
    print(ymax)
    for i in range(0,n):
        if len(X_train[i])>0:
            x = X_train[i][:,0]
            y = X_train[i][:,1]
            X_train_features_cc1[i] = sum(x*(y-x))
            X_train_features_cc2[i] = sum((ymax - y)*(y-x))
            X_train_features_cc3[i] = sum(x**2*(y-x)**4)
            X_train_features_cc4[i] = sum((ymax-y)**2*(y-x)**4)
        else:
            X_train_features_cc1[i] = 0
            X_train_features_cc2[i] = 0
            X_train_features_cc3[i] = 0
            X_train_features_cc4[i] = 0

    n = len(X_test)
    X_test_features_cc1 = np.zeros(n)
    X_test_features_cc2 = np.zeros(n)
    X_test_features_cc3 = np.zeros(n)
    X_test_features_cc4 = np.zeros(n)
    ymax = 0
    for i in range(0,n):
        if len(X_test[i])>0:
            b = np.max(X_test[i][:,1])
        else:
            b = ymax
        if ymax < b:
            ymax = b
        else:
            ymax = ymax
    for i in range(0,n):
        if len(X_test[i])>0:
            x = X_test[i][:,0]
            y = X_test[i][:,1]
            X_test_features_cc1[i] = sum(x*(y-x))
            X_test_features_cc2[i] = sum((ymax - y)*(y-x))
            X_test_features_cc3[i] = sum(x**2*(y-x)**4)
            X_test_features_cc4[i] = sum((ymax-y)**2*(y-x)**4)
        else:
            X_test_features_cc1[i] = 0
            X_test_features_cc2[i] = 0
            X_test_features_cc3[i] = 0
            X_test_features_cc4[i] = 0
    print("Total Time (Carlsson Coordinates): ", time.time()-start)
    return X_train_features_cc1, X_train_features_cc2, X_train_features_cc3, X_train_features_cc4, X_test_features_cc1, X_test_features_cc2, X_test_features_cc3, X_test_features_cc4

def landscape_features(X_train, X_test, num_landscapes=5, resolution=100):
    landscapes = Landscape(num_landscapes, resolution)
    lr = landscapes.fit(X_train)
    X_train_features = lr.transform(X_train)
    X_test_features = lr.transform(X_test)
    return X_train_features, X_test_features

def persistence_image_features(X_train, X_test, pixels=[30,30], spread=1):
    pim = PersImage(pixels=pixels, spread=spread)
    imgs_train = pim.transform(X_train)
    X_train_features = np.array([img.flatten() for img in imgs_train])
    pim = PersImage(pixels=pixels, spread=spread)
    imgs_test = pim.transform(X_test)
    X_test_features = np.array([img.flatten() for img in imgs_test])
    return X_train_features, X_test_features

def fast_hk(dgm0,dgm1,sigma=.4):
    dist = DistanceMetric.get_metric('euclidean')
    dist1 = (dist.pairwise(dgm0,dgm1))**2
    Qc = dgm1[:,1::-1]
    dist2 = (dist.pairwise(dgm0,Qc))**2
    exp_dist1 = sum(sum(np.exp(-dist1/(8*sigma))))
    exp_dist2 = sum(sum(np.exp(-dist2/(8*sigma))))
    hk = (exp_dist1-exp_dist2)/(8*np.pi*sigma)
    return(hk)

def heat_kernel_approx(dgm0, dgm1, sigma=.4):
    return np.sqrt(fast_hk(dgm0, dgm0, sigma) + fast_hk(dgm1, dgm1, sigma) - 2*fast_hk(dgm0, dgm1, sigma))

def fast_kernel_features(train, test, s):
    n_train = len(train)
    n_test = len(test)
    X_train_features = np.zeros((n_train, n_train))
    X_test_features = np.zeros((n_test, n_train))
    
    start = time.time()
    for i in range(0,n_train):
        if i % 5 == 0:
            print("Iteration: ", i)
            print("Total Time: ", time.time() - start)
            print("Iterations left: " , n_train - i)
        for j in range(0,i):
            dgm0 = train[i]
            dgm1 = train[j]
            hka = heat_kernel_approx(dgm0, dgm1, sigma = s) 
            X_train_features[i,j] = hka
            X_train_features[j,i] = hka
        for j in range(0,n_test):
            dgm0 = train[i]
            dgm1 = test[j]
            hka = heat_kernel_approx(dgm0, dgm1, sigma = s)        
            X_test_features[j,i] = hka

    print("Total Time (Kernel): ", time.time()-start)
    return X_train_features, X_test_features

    