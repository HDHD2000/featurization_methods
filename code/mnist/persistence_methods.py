import numpy as np
from gudhi.representations.vector_methods import Landscape, PersistenceImage, Silhouette, Entropy
from gudhi.representations.kernel_methods import SlicedWassersteinKernel, PersistenceWeightedGaussianKernel, PersistenceScaleSpaceKernel, PersistenceFisherKernel

def landscape_features(X_train, X_test, num_landscapes=5, resolution=100):
    landscapes = Landscape(num_landscapes, resolution)
    lr = landscapes.fit(X_train)
    X_train_features = lr.transform(X_train)
    X_test_features = lr.transform(X_test)
    return X_train_features, X_test_features

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