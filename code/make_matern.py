import numpy as np
from ripser import Rips

rips = Rips()
data = np.random.random((1000,10))
diagrams = rips.fit_transform(data)
rips.plot(diagrams)