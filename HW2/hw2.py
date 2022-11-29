#!/usr/bin/env python
# coding: utf-8

# # HW 2 - Discrimination by Regression
# ## Murat Han Aydoğan
# ### 64756

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# ## Importing Data

# In[2]:


# read data into memory
data_set = np.genfromtxt("hw02_data_points.csv", delimiter=",")
data_set_y = np.genfromtxt("hw02_class_labels.csv")

# get X
X = data_set[:10000].astype(float)
X_test = data_set[10000:].astype(float)

# get number of samples
N = X.shape[0]
N_test = X_test.shape[0]
# get number of features
D = X.shape[1]


# In[3]:


print(N, N_test, D)


# In[4]:


#get y values
y_truth = np.transpose(np.array([data_set_y[:10000].astype(int)]))
y_test = np.transpose(np.array([data_set_y[10000:].astype(int)]))

# get number of classes
K = np.max(y_truth)
print(K)

# one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth[:, 0] - 1] = 1

Y_test = np.zeros((N_test, K)).astype(int)
Y_test[range(N_test), y_test[:, 0] - 1] = 1

print(Y_truth)
print(Y_test)


# ## Sigmoid Function

# $\textrm{sigmoid}(\boldsymbol{w}^{\top} \boldsymbol{x} + w_{0}) = \dfrac{1}{1 + \exp\left[-(\boldsymbol{w}^{\top} \boldsymbol{x} + w_{0})\right]}$

# In[5]:


# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


# ## Gradient Functions

# \begin{align*}
# \dfrac{\partial \textrm{Error}}{\partial \boldsymbol{w}_{c}} &= -\sum\limits_{i = 1}^{N} (y_{ic} - \widehat{y}_{ic})(\widehat{y}_{ic})(1 - \widehat{y}_{ic})\boldsymbol{x}_{i} \\
# \dfrac{\partial \textrm{Error}}{\partial w_{c0}} &= -\sum\limits_{i = 1}^{N} (y_{ic} - \widehat{y}_{ic})(\widehat{y}_{ic})(1- \widehat{y}_{ic}) 
# \end{align*}

# In[6]:


# define the gradient functions
def gradient_W(X, Y_truth, Y_predicted):
    y = ((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted))
    return(np.asarray([-np.matmul(y[:, c], X) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    y = ((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted))
    return(-np.sum(y, axis = 0))


# ## Algorithm Parameters

# In[7]:


# set learning parameters
eta = 0.00001
iteration_count = 1000


# In[8]:


W_data_set = np.genfromtxt("hw02_W_initial.csv", delimiter=",")
w0_data_set = np.genfromtxt("hw02_w0_initial.csv", delimiter=",")

W = W_data_set.astype(float)

w0 = w0_data_set.astype(float)


# ## Iterative Algorithm

# $\textrm{Error} = 0.5 \sum\limits_{i = 1}^{N} \sum\limits_{c = 1}^{K} \left[ y_{ic}- \hat{y}_{ic} \right]^2$

# In[9]:


# learn W and w0 using gradient descent
iteration = 1
objective_values = []
while True:
    Y_predicted = sigmoid(X, W, w0)

    objective_values = np.append(objective_values, 0.5 * np.sum((Y_truth - Y_predicted)**2))

    W_old = W
    w0_old = w0

    W = W_old - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0_old - eta * gradient_w0(Y_truth, Y_predicted)

    if iteration == iteration_count:
        break

    iteration = iteration + 1
print(W)
print(w0)


# In[10]:


# plot objective function during iterations
plt.figure(figsize = (8, 4))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# ## Training Performance

# In[11]:


# calculate confusion matrix
y_predicted = np.array([np.argmax(Y_predicted, axis = 1) + 1])
confusion_matrix = pd.crosstab(y_predicted[0], y_truth.T[0],
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])

print(confusion_matrix)


# # Testing Data

# In[12]:


# learn W and w0 using gradient descent
iteration = 1
objective_values_test = []
while True:
    Y_predicted_test = sigmoid(X_test, W, w0)

    objective_values_test = np.append(objective_values_test, 0.5 * np.sum((Y_test - Y_predicted_test)**2))

    W_old = W
    w0_old = w0

    W = W_old - eta * gradient_W(X_test, Y_test, Y_predicted_test)
    w0 = w0_old - eta * gradient_w0(Y_test, Y_predicted_test)

    if iteration == iteration_count:
        break

    iteration = iteration + 1
print(W)
print(w0)


# In[13]:


# plot objective function during iterations
plt.figure(figsize = (8, 4))
plt.plot(range(1, iteration + 1), objective_values_test, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[14]:


# calculate confusion matrix
y_predicted_test = np.array([np.argmax(Y_predicted_test, axis = 1) + 1])
confusion_matrix_test = pd.crosstab(y_predicted_test[0], y_test.T[0],
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])

print(confusion_matrix_test)


# In[ ]:




