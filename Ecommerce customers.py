#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df = pd.read_csv('Ecommerce Customers.csv')


# In[16]:


df.head()


# In[17]:


df.info()


# In[18]:


df.describe()


# In[19]:


df.columns


# In[22]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)


# In[24]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)


# In[25]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=df)


# In[26]:


sns.pairplot(df)


# In[27]:


# Length of Membership 


# In[28]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)


# In[29]:


sns.lmplot(x='Time on App',y='Yearly Amount Spent',data=df)


# In[35]:


y= df['Yearly Amount Spent']


# In[37]:


X= df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=101)


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


lm= LinearRegression()


# In[42]:


lm.fit(X_train,y_train)


# In[44]:


predictions = lm.predict(X_test)


# In[45]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[49]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('RTS:',metrics.r2_score(y_test, predictions))


# In[52]:


sns.distplot((y_test-predictions),bins=50);


# In[ ]:




