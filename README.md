# featurization_methods

This repository contains the code used in the comparison of featurization methods on different data sets.
These data sets are the ones studied in the master thesis and include sampled points from the torus and the sphere, orbits of a discrete dynamical system, forms of a protein-binding molecule, the MNIST dataset and time series datasets from the UCR archive.
For each data set, we compare eight different featurization methods, of which are four kernel methods and six vectorization methods.
The four kernel methods used here are: the Sliced Wasserstein kernel (SWK), the Persistence Weighted Gaussian kernel (PWGK), the Persistence Scale-Space kernel (PSSK) and the Persistence Fisher kernel (PFK).
The six vectorization methods used here are: the Persistence Landscape, the Persistence Image, the Persistence Silhouette, the Persistence Entropy, the Adcock-Carlsson coordinates and the Tropical coordinates.

The goal is to see how these featurization methods behave in terms of computational efficiency and classification accuracy in the case of each data set.
For all data sets, the code constructing the persistence diagrams, applying the featurization methods and using the SVC classification method is heavily inspired from the GUDHI Jupyter textbooks.
Their textbooks can be found on their github page: https://github.com/GUDHI/TDA-tutorial.
Our code for these classification tasks can be found in the 'code' folder.
This folder contains five folders corresponding to each data set we are studying.
Each of these folders contains Python code than can be implemented to study the behaviour of these featurization methods.

The parameters used for the classification tasks have been added as comments at the end of each code file.
The authors are aware that these are not the best parameter choices, but offer nonetheless a rough comparison between the different featurization methods.

The folder 'graphical_results' contains some images which were used in the master thesis.
These images include graphical representations of the vectorization methods, plots of persistence diagrams and illustrations of the data sets, such as the sampled points from the torus and the sphere, the MNIST dataset or the orbits of the discrete dynamical system.
