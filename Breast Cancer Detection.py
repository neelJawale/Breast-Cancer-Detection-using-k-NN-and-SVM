#!/usr/bin/env python
# coding: utf-8

# In[115]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


# In[116]:


import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd


# In[117]:


# Load the dataset and create a dataframe.
data = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
features = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(data, names=features)
print(df)


# In[118]:


# Preprocess the dataset
df.replace('?',-99999, inplace=True) # Replace the missing or invalid data with a placeholder value to keep data consistent.
print(df.axes)
df.drop(['id'], axis=1, inplace=True)


# In[119]:


# Explore the nature of the data
print(df.loc[698])
print(df.shape)


# In[120]:


print(df.describe())


# In[121]:


# Histogram Visualization of the data
df.hist(figsize = (10, 10))
plt.show()


# In[122]:


# To see the inter-dependencies of the variables, use scatter plot
scatter_matrix(df, figsize = (18,18))
plt.show()


# In[123]:


# Create X and Y datasets for training
X = np.array(df.drop(['class'],axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


# In[124]:


seed = 12 # Define seed for reproducibility of the results during validation
scoring = 'accuracy'


# In[125]:


# Define models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 3)))
models.append(('SVM', SVC()))

results = []
names = []

# Evaluate each model
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring= scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[129]:


# Make predictions on test/validation dataset
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
print(predictions)


# In[127]:


clf = SVC()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Use SVM to test an individual example set
example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

