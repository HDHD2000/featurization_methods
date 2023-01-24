from sktime.datasets import load_UCR_UEA_dataset
import numpy as np
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt

data_series, labels = load_UCR_UEA_dataset(name="Lightning7")

num_train = 363
num_test = 24

data_series = data_series.to_numpy()
data_series = data_series.flatten()

num_data_series = np.shape(data_series)[0]

for j in range(20):
    plt.plot(data_series[j])
    plt.show()