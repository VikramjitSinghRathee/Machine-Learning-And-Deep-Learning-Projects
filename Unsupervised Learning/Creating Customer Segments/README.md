## customer_segments.ipynb

In this project, dataset containing data on various customers' annual spending amounts (reported in monetary units) of diverse product categories for internal structure was analysed. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.

Feature scaling and outlier removal was utilized through sklearn library to pre-process the data. Applied PCA (Principal Component Analysis) dimensionality reduction on the dataset and then utilized K-means clustering on the PCA transformed data to create customer segments (clusters corresponding to particular establishments) and then finally gave conclusion about delivery service to particular establishments.
