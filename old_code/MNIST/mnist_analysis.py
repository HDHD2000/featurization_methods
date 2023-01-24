import numpy as np
from sklearn.preprocessing   import MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC
import gudhi as gd
import gudhi.representations
from keras.datasets import mnist

path_file = "./MNIST_PD/"

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_dgms = [np.load(path_file + 'PD_train_X[' + str(j) + '].npy') for j in range(10000)]
test_dgms = [np.load(path_file + 'PD_test_X[' + str(i) + '].npy') for i in range(2000)]

np.load = np_load_old

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_y = train_y[:10000]
test_y = test_y[:2000]

pipe = Pipeline([("Separator", gd.representations.DiagramSelector(limit=np.inf, point_type="finite")),
                     ("Scaler",    gd.representations.DiagramScaler(scalers=[([0,1], MinMaxScaler())])),
                     ("TDA",       gd.representations.Landscape()),
                     ("Estimator", SVC())])

model = pipe.fit(train_dgms, train_y)
print(model.score(train_dgms, train_y))
print(model.score(test_dgms,  test_y))
