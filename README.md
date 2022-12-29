# Breast-Cancer-Detection-using-k-NN-and-SVM
A breast cancer detection program that uses supervised learning based ML algorithms, viz., k-NN and SVM to classify the cancer type as benign and malignant. 

## The upshot of SVM:
- The goal of an SVM is to find the hyperplane in a high-dimensional space that maximally separates the different classes. In the case of classification, the SVM algorithm tries to find the hyperplane that maximally separates the data points of different classes. 
- The hyperplane is chosen such that the distance from it to the nearest data points of either class is maximized. These data points are called support vectors and the hyperplane is called the maximum-margin hyperplane. 
- One of the key benefits of SVM algorithms is that they can perform well even when the data is not linearly separable by using a kernel trick to transform the data into a higher-dimensional space where it becomes linearly separable.

## The upshot of k-NN:
- The algorithm works by storing all available cases and classifying new cases based on a similarity measure (e.g., distance functions). 
- In the case of classification, the k-NN algorithm assigns a new data point to the class that is most common among its k nearest neighbors, where k is a positive integer that is specified by the user. 
- One of the main advantages of the k-NN algorithm is that it is easy to understand and implement, and it requires little training data. It is also resistant to overfitting, as it makes predictions based on the majority class among the k nearest neighbors rather than making a decision based on a complex model.
