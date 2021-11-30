#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# # Importing the dataset

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Credit card fraud detection/creditcard.csv')


# In[3]:


#print fisrt 5 rows of the dataset
df.head()


# In[4]:


#print the last 5 rows of the dataset
df.tail()


# In[5]:


#shape of the dataset
df.shape


# In[6]:


#dataset information
df.info()


# In[7]:


#distribution of class values
df['Class'].value_counts()


# This dataset is highly unbalanced as the number of fraud cases are much much smaller than legit transactions.
# 
# 0 ----> legit transactions
# 1 ----> fraud transactions

# In[8]:


#separating the data for analysis
legit = df[df.Class == 0]
fraud = df[df.Class == 1]


# In[9]:


print(legit.shape, fraud.shape)


# In[10]:


#stastical measures of the dataset
legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


#compare the values for both transactions
df.groupby('Class').mean()


# # Under Sampling

# Build a sample dataset containing similar distributions of normal and fraudlent transactions.
# 
# No. of fraudlent transactions ---> 492

# In[13]:


legit_sample = legit.sample(n = 492)


# In[14]:


#concatenate two dataframes
new_df = pd.concat([legit_sample, fraud], axis = 0)


# In[15]:


#print first 5 rows
new_df.head()


# In[16]:


#print last 5 rows
new_df.tail()


# In[17]:


#shape of new_df
new_df.shape


# In[18]:


#distribution of class values
new_df['Class'].value_counts()


# In[19]:


#compare the values for both transactions
new_df.groupby('Class').mean()


# Splitting the data into features and targets

# In[20]:


X = new_df.drop('Class', axis = 1)
Y = new_df['Class']


# In[21]:


print(X.shape, Y.shape)


# # Train and test data and model evaluation

# In[22]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, stratify = Y, random_state = 2)


# In[23]:


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# Model evaluation
# 
# Logistic regression

# In[24]:


model = LogisticRegression()


# In[25]:


#training the model
model.fit(x_train, y_train)


# Model evaluation
# 
# Accuracy score

# In[26]:


#on training data
training_prediction = model.predict(x_train)

training_accuracy = accuracy_score(training_prediction, y_train)
print('THE TRAINING ACCURACY IS :', training_accuracy)


# In[27]:


#on testing data
testing_prediction = model.predict(x_test)

testing_accuracy = accuracy_score(testing_prediction, y_test)
print('THE TRAINING ACCURACY IS :', testing_accuracy)


# # Building a predictive system

# In[29]:


model_input = input()

input_list = [float(i) for i in model_input.split(',')]

input_array = np.asarray(input_list)

reshaped_array = input_array.reshape(1, -1)

prediction = model.predict(reshaped_array)
if prediction == 1:
    print('\nTHIS IS A CASE OF FRAUDLENT TRANSACTION')
else:
    print('THE IS A LEGIT TRANSACTION')


# In[ ]:




