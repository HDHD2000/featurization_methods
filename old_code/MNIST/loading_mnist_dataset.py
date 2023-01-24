from keras.datasets import mnist
import gudhi as gd
import numpy as np

path_file = "./MNIST_PD/"

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X[:10000]
train_y = train_y[:10000]
test_X = test_X[:2000]
test_y = test_y[:2000]

#for i in range(10000):
 #   cc_image = gd.CubicalComplex(top_dimensional_cells = train_X[i])
  #  dgm = cc_image.persistence()
   # np.save(path_file + 'PD_train_X[' + str(i) + '].npy', dgm)

for j in range(2000):
    cc_image = gd.CubicalComplex(top_dimensional_cells = test_X[j])
    dgm = cc_image.persistence()
    np.save(path_file + 'PD_test_X[' + str(j) + '].npy', dgm)