#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv('USA_Housing.csv')


# In[7]:


df.head()


# In[8]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.columns


# # EDA

# In[36]:


sns.pairplot(USAhousing)


# In[14]:


sns.distplot(USAhousing['Price'])


# In[15]:


sns.heatmap(USAhousing.corr())


# In[16]:


df.corr()


# In[17]:


sns.heatmap(USAhousing.isnull())


# # Training a Linear Regression Model

# In[18]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# # Train test Split

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[21]:


y_test.shape


# # Creating and training model

# In[22]:


import sklearn

from sklearn.linear_model import LinearRegression


# In[23]:


lm = sklearn.linear_model.LinearRegression()


# In[25]:


lm.fit(X_train,y_train)


# # Predictions

# In[26]:


predictions = lm.predict(X_test)


# In[27]:


predictions


# In[29]:


plt.scatter(y_test,predictions)


# # residual histogram 

# In[30]:


sns.distplot((y_test-predictions),bins=50);


# # Metrics

# In[31]:


from sklearn import metrics


# In[34]:


print('MAE:',metrics.mean_absolute_error(y_test, predictions))
print('MSE:',metrics.mean_squared_error(y_test, predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




