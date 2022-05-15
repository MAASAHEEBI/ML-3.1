#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt', index_col='Unnamed: 0')
df.head()


# # Data Manipulation

# In[3]:


df.shape


# In[4]:


df.isna().sum()


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


df.rename(columns={'Living.Room':'Livingroom'}, inplace=True)


# In[10]:


df.head()


# # visualization

# In[12]:


from matplotlib import pyplot as plt
import seaborn as sns


# #price per sqft

# In[17]:


sns.scatterplot(x='Sqft',y='Price',data=df)


# In[19]:


sns.scatterplot(x='Bedroom',y='Price',data=df)


# In[20]:


y=df[['Price']]


# # model selection

# In[25]:


feature_data=df[['Sqft', 'Floor', 'TotalFloor', 'Bedroom', 'Livingroom', 'Bathroom']]


# In[26]:


feature_data


# In[27]:


from sklearn.model_selection import train_test_split
trainX,testX, trainY,testY = train_test_split(feature_data, y)


# In[28]:


trainX.shape


# In[29]:


testX.shape


# In[30]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[31]:


lr = LinearRegression(normalize=True)
lr.fit(trainX,trainY)


# In[32]:


lr.coef_


# In[33]:


testX[:5]


# In[34]:


testY[:5]


# In[35]:


lr.predict(testX[:5])


# In[36]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[37]:


pred = lr.predict(testX)
mean_absolute_error(y_pred=pred, y_true=testY)


# In[38]:


from sklearn.linear_model import Ridge,Lasso


# In[39]:


ridge = Ridge(alpha=1000)
lasso = Lasso(alpha=1000)


# In[40]:


ridge.fit(trainX,trainY)
lasso.fit(trainX,trainY)


# In[41]:


pred = ridge.predict(testX)
pred = lasso.predict(testX)


# In[42]:


mean_absolute_error(y_pred=pred, y_true=testY)


# In[ ]:




