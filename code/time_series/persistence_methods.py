import numpy as np
from sklearn.preprocessing import scale

def carlsson_coordinates(X):
    n = len(X)
    X_cc1 = np.zeros(n)
    X_cc2 = np.zeros(n)
    X_cc3 = np.zeros(n)
    X_cc4 = np.zeros(n)
    ymax = 0
    for i in range(0,n):
        if len(X[i])>0:
            b = np.max(X[i][:,1])
        else:
            b = ymax
        if ymax < b:
            ymax = b
        else:
            ymax = ymax
    for i in range(0,n):
        if len(X[i])>0:
            x = X[i][:,0]
            y = X[i][:,1]
            X_cc1[i] = sum(x*(y-x))
            X_cc2[i] = sum((ymax - y)*(y-x))
            X_cc3[i] = sum(x**2*(y-x)**4)
            X_cc4[i] = sum((ymax-y)**2*(y-x)**4)
        else:
            X_cc1[i] = 0
            X_cc2[i] = 0
            X_cc3[i] = 0
            X_cc4[i] = 0
    return np.column_stack((scale(X_cc1), scale(X_cc2), scale(X_cc3), scale(X_cc4)))

def tropical_coordinates(X):
    n = len(X)
    X_tc1 = np.zeros(n)
    X_tc3 = np.zeros(n)
    X_tc4 = np.zeros(n)
    X_tc5 = np.zeros(n)
    X_tc7 = np.zeros(n)
    for i in range(0,n):
        m = len(X[i])
        if m>0:
            sum_max2 = 0
            sub = np.zeros(m)
            x = X[i][:,0]
            y = X[i][:,1]
            X_tc1[i] = max(y-x)
            X_tc3[i] = sum(y-x)
            for j in range(0,m):
                sub[j] = min(28*(y[j] - x[j]), x[j])
            X_tc7[i] = sum(sub)
            max_sub = max(sub + y-x)
            X_tc4[i] = sum(max_sub - sub)
            for q in range(0,m):
                if q>0:
                    for k in range(0,q-1):
                        sum2 = y[q] - x[q] + y[k] - x[k]
                        if sum2>sum_max2:
                            sum_max2 = sum2
                    X_tc5[i] = sum_max2
                else:
                    X_tc5[i] = 0
        else:
            X_tc1[i] = 0
            X_tc3[i] = 0
            X_tc7[i] = 0
            X_tc4[i] = 0
            X_tc5[i] = 0
            
    return np.column_stack((scale(X_tc1),scale(X_tc3),scale(X_tc4),scale(X_tc5),scale(X_tc7)))