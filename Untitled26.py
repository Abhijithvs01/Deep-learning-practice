#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install kagglehub')


# In[4]:


import kagglehub


# In[22]:


path = kagglehub.dataset_download("meet3010/insurance-dataset")
print(path)


# In[24]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


import os


# In[26]:


os.listdir(path)


# In[27]:


print(path)


# In[29]:


data = pd.read_csv(path + "/insurance_data.csv")


# In[30]:


data


# In[31]:


data.head()


# In[34]:


get_ipython().system('pip install scikit-learn')


# In[36]:


import sklearn
from sklearn.model_selection import train_test_split


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(data[['age']],data.bought_insurance,test_size=0.2,random_state = 25)


# In[40]:





# In[41]:


len(X_train)


# In[53]:


X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100
X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100


# In[57]:


model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,), activation = 'sigmoid', kernel_initializer = 'ones', bias_initializer = 'zeros')
])


# In[60]:


model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics=['accuracy']) 


# model.fit(X_train_scaled, y_train, epochs = 5000)

# In[62]:


model.fit(X_train_scaled, y_train, epochs = 100)


# In[64]:


X_test_scaled


# In[63]:


model.predict(X_test_scaled)


# In[65]:


y_test


# In[66]:


coef , intercept = model.get_weights()


# In[67]:


coef, intercept


# In[75]:


def sigmoid(x):
    import math 
    return 1 / (1+math.exp(-x))


# In[80]:


def pred(age):
    w_s = coef[0][0]*age + intercept[0]
    return sigmoid(w_s)


# In[81]:


pred(.47,)


# In[78]:


print(coef)
print(intercept)
print(type(coef))
print(type(intercept))


# In[83]:


def log_loss(y_true,y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


# In[104]:


def sigmoid_numpy(X):
    return 1/(1+np.exp(-X))


# In[110]:


def gradeint_descent(age,y_true,epochs):
    w1 = w2 = 1
    bias = 0
    rate = 0.5
    n = len(age)
    for i in range(epochs):
        weighted_sum = w1 * age + bias
        y_predicted = sigmoid_numpy(weighted_sum)

        loss = log_loss(y_true, y_predicted)

        w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true))

        bias_d = np.mean(y_predicted-y_true)

        w1 = w1 - rate * w1d
        bias = bias - rate * bias_d

        accuracy = np.mean((y_predicted >= 0.5) == y_true) * 100

        print(f'Epoch:{i}, w1:{w1}, bias:{bias}, loss:{loss},accuracy:{accuracy}')



    return w1, w2, bias 




# In[111]:


gradeint_descent(X_train_scaled['age'],y_train,1000)


# In[112]:


git init


# In[ ]:




