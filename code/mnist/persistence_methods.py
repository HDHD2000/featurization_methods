import numpy as np
from gudhi.representations.vector_methods import Landscape, PersistenceImage, Silhouette, Entropy
from gudhi.representations.kernel_methods import SlicedWassersteinKernel, PersistenceWeightedGaussianKernel, PersistenceScaleSpaceKernel, PersistenceFisherKernel

def landscape_features(X_train, X_test, num_landscapes=5, resolution=100):
    landscapes = Landscape(num_landscapes, resolution)
    lr = landscapes.fit(X_train)
    X_train_features = lr.transform(X_train)
    X_test_features = lr.transform(X_test)
    return X_train_features, X_test_features

def carlsson_coordinates(X_train, X_test):
    n = len(X_train)
    X_train_features_cc1 = np.zeros(n)
    X_train_features_cc2 = np.zeros(n)
    X_train_features_cc3 = np.zeros(n)
    X_train_features_cc4 = np.zeros(n)
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
    return X_train_features_cc1, X_train_features_cc2, X_train_features_cc3, X_train_features_cc4, X_test_features_cc1, X_test_features_cc2, X_test_features_cc3, X_test_features_cc4

def tropical_coordinates(X_train, X_test):
    n = len(X_train)
    X_train_tc1 = np.zeros(n)
    X_train_tc3 = np.zeros(n)
    X_train_tc4 = np.zeros(n)
    X_train_tc5 = np.zeros(n)
    X_train_tc7 = np.zeros(n)
    for i in range(0,n):
        m = len(X_train[i])
        if m>0:
            sum_max2 = 0
            sub = np.zeros(m)
            x = X_train[i][:,0]
            y = X_train[i][:,1]
            X_train_tc1[i] = max(y-x)
            X_train_tc3[i] = sum(y-x)
            for j in range(0,m):
                sub[j] = min(28*(y[j] - x[j]), x[j])
            X_train_tc7[i] = sum(sub)
            max_sub = max(sub + y-x)
            X_train_tc4[i] = sum(max_sub - sub)
            for q in range(0,m):
                if q>0:
                    for k in range(0,q-1):
                        sum2 = y[q] - x[q] + y[k] - x[k]
                        if sum2>sum_max2:
                            sum_max2 = sum2
                    X_train_tc5[i] = sum_max2
                else:
                    X_train_tc5[i] = 0
        else:
            X_train_tc1[i] = 0
            X_train_tc3[i] = 0
            X_train_tc7[i] = 0
            X_train_tc4[i] = 0
            X_train_tc5[i] = 0
            
    n = len(X_test)
    X_test_tc1 = np.zeros(n)
    X_test_tc3 = np.zeros(n)
    X_test_tc4 = np.zeros(n)
    X_test_tc5 = np.zeros(n)
    X_test_tc7 = np.zeros(n)
    for i in range(0,n):
        m = len(X_test[i])
        if m>0:
            sum_max2 = 0
            sub = np.zeros(m)
            x = X_test[i][:,0]
            y = X_test[i][:,1]
            X_test_tc1[i] = max(y-x)
            X_test_tc3[i] = sum(y-x)
            for j in range(0,m):
                sub[j] = min(28*(y[j] - x[j]), x[j])
            X_test_tc7[i] = sum(sub)
            max_sub = max(sub + y-x)
            X_test_tc4[i] = sum(max_sub - sub)
            for q in range(0,m):
                if q>0:
                    for k in range(0,q-1):
                        sum2 = y[q] - x[q] + y[k] - x[k]
                        if sum2>sum_max2:
                            sum_max2 = sum2
                    X_test_tc5[i] = sum_max2
                else:
                    X_test_tc5[i] = 0
        else:
            X_test_tc1[i] = 0
            X_test_tc3[i] = 0
            X_test_tc7[i] = 0
            X_test_tc4[i] = 0
            X_test_tc5[i] = 0
            
    return X_train_tc1,X_train_tc3,X_train_tc4,X_train_tc5,X_train_tc7, X_test_tc1,X_test_tc3,X_test_tc4,X_test_tc5,X_test_tc7

def persistence_image_features(X_train, X_test, pixels=[30,30], bandwidth=1):
    pim = PersistenceImage(bandwidth = bandwidth, weight = lambda x: 1, resolution = pixels)
    pi = pim.fit(X_train)    
    imgs_train = pi.transform(X_train)
    X_train_features = np.array([img.flatten() for img in imgs_train])
    imgs_test = pi.transform(X_test)
    X_test_features = np.array([img.flatten() for img in imgs_test])
    return X_train_features, X_test_features

def swk_features(X_train, X_test, num_direc = 10, bandwidth = 1):
    swk = SlicedWassersteinKernel(num_directions = num_direc, bandwidth = bandwidth)
    sw = swk.fit(X_train)
    X_train_features = sw.transform(X_train)
    X_test_features = sw.transform(X_test)
    return X_train_features, X_test_features

def silhouette_features(X_train, X_test, resolution = 100):
    silhouette = Silhouette(resolution = resolution)
    sil = silhouette.fit(X_train)
    X_train_features = sil.transform(X_train)
    X_test_features = sil.transform(X_test)
    return X_train_features, X_test_features

def pwgk_features(X_train, X_test, bandwidth = 1):
    pwgk = PersistenceWeightedGaussianKernel(weight = lambda x: np.arctan(x[1]-x[0]), bandwidth = bandwidth)
    pwg = pwgk.fit(X_train)
    X_train_features = pwg.transform(X_train)
    X_test_features = pwg.transform(X_test)
    return X_train_features, X_test_features

def pssk_features(X_train, X_test, bandwidth = 1):
    pssk = PersistenceScaleSpaceKernel(bandwidth = bandwidth)
    pss = pssk.fit(X_train)
    X_train_features = pss.transform(X_train)
    X_test_features = pss.transform(X_test)
    return X_train_features, X_test_features

def pfk_features(X_train, X_test, bandwidth = 1, bandwidth_fisher = 1):
    pfk = PersistenceFisherKernel(bandwidth_fisher = bandwidth_fisher, bandwidth = bandwidth)
    pf = pfk.fit(X_train)
    X_train_features = pf.transform(X_train)
    X_test_features = pf.transform(X_test)
    return X_train_features, X_test_features   

def entropy_features(X_train, X_test, resolution = 100):
    entropy = Entropy(mode = 'vector', resolution = resolution)
    entr = entropy.fit(X_train)
    X_train_features = entr.transform(X_train)
    X_test_features = entr.transform(X_test)
    return X_train_features, X_test_features 