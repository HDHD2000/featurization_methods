import numpy as np

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
