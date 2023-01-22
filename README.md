# featurization_methods

This repository contains the code used in the comparison of numerous featurization methods on several data sets.
These data sets are the ones studied in the master thesis and include sampled points from the torus and the sphere, orbits of a discrete dynamical system, forms of a protein-binding molecule, the MNIST dataset and time series datasets from the UCR archive.
For each faeturization method, we compare eight different featurization methods, four kernel methods and four vectorization methods.
The four kernel methods used here are: the Sliced Wasserstein kernel (SWK), the Persistence Weighted Gaussian kernel (PWGK), the Persistence Scale-Space kernel (PSSK) and the Persistence Fisher kernel (PFK).
The four vectorization methods used here are: the Persistence Landscape, the Persistence Image, the Persistence Silhouette and the Persistence Entropy.

The goal is to see how these featurization methods behave in terms of computational efficiency and classification accuracy in the case of each data set.
For all data sets, the code used to construct the persistence diagrams from the data set, apply the featurization method and classify using the SVC machien learning method is heavily inspired from the GUDHI textbooks.
