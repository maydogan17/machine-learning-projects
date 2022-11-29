#!/usr/bin/env python
# coding: utf-8

# # HW 01 - : Naive Bayes Classifier
# ## Murat Han Aydoğan
# ### 64756

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd


# ## Importing Data

# In[2]:


data_set_x = np.genfromtxt("hw01_data_points.csv", dtype="S1" , delimiter=",")
data_set_y = np.genfromtxt("hw01_class_labels.csv")

training_y = data_set_y[0:300].astype(int)
training_x = data_set_x[0:300].astype(str)

test_y = data_set_y[300:400].astype(int)
test_x = data_set_x[300:400].astype(str)

training_N = len(training_x)
testing_N = len(test_x)

seq_len = 7

training_group_1 = len(training_x[training_y == 1])
training_group_2 = training_N - training_group_1


# ## Parameter Estimation

# In[3]:


#pAcd =np.array([ [(np.count_nonzero(np.transpose(training_x[training_y == j+1])[k] == 'A') / training_group_1) for k in range(seq_len)] for j in range(2)])
#pCcd =np.array([ [(np.count_nonzero(np.transpose(training_x[training_y == j+1])[k] == 'C') / training_group_1) for k in range(seq_len)] for j in range(2)])
#pGcd =np.array([ [(np.count_nonzero(np.transpose(training_x[training_y == j+1])[k] == 'G') / training_group_1) for k in range(seq_len)] for j in range(2)])
#pTcd =np.array([ [(np.count_nonzero(np.transpose(training_x[training_y == j+1])[k] == 'T') / training_group_1) for k in range(seq_len)] for j in range(2)])

pAcd = np.append(
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 1])[k] == 'A') / training_group_1) for k in range(seq_len)])],
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 2])[k] == 'A') / training_group_2) for k in range(seq_len)])], axis=0)
pCcd = np.append(
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 1])[k] == 'C') / training_group_1) for k in range(seq_len)])],
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 2])[k] == 'C') / training_group_2) for k in range(seq_len)])], axis=0)
pGcd = np.append(
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 1])[k] == 'G') / training_group_1) for k in range(seq_len)])],
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 2])[k] == 'G') / training_group_2) for k in range(seq_len)])], axis=0)
pTcd = np.append(
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 1])[k] == 'T') / training_group_1) for k in range(seq_len)])],
         [np.array([(np.count_nonzero(np.transpose(training_x[training_y == 2])[k] == 'T') / training_group_2) for k in range(seq_len)])], axis=0)

class_priors = np.array([training_group_1/training_N, training_group_2/training_N])


# In[4]:


print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)
print(class_priors)


# ## Scoring Function

# In[5]:


def scoring_func(x):
    g1 = 1
    g2 = 1
    for i in range(len(x)):
        if(x[i] == 'A'):
            g1 *= pAcd[0][i]
            g2 *= pAcd[1][i]
        elif(x[i] == 'C'):
            g1 *= pCcd[0][i]
            g2 *= pCcd[1][i]
        elif(x[i] == 'G'):
            g1 *= pGcd[0][i]
            g2 *= pGcd[1][i]
        else:
            g1 *= pTcd[0][i]
            g2 *= pTcd[1][i]
    g1 = np.log(g1) + np.log(class_priors[0])
    g2 = np.log(g2) + np.log(class_priors[1])
    
    return 1 if g1 > g2 else 2

predicted_training_y = np.array([scoring_func(training_x[k]) for k in range(len(training_x))])
predicted_test_y = np.array([scoring_func(test_x[k]) for k in range(len(test_x))])


# ## Confusion Matrix

# In[6]:


confusion_train = pd.crosstab(predicted_training_y, training_y, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_train)


# In[7]:


confusion_test = pd.crosstab(predicted_test_y, test_y, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_test)


# In[ ]:




