#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### EDA

# In[56]:


df=pd.read_csv('titanic_train.csv')


# In[57]:


df.head()


# In[58]:


sns.countplot(data=df,x='Survived',hue='Sex')


# In[59]:


df['Survived'].value_counts().plot(kind='bar')


# In[60]:


pd.get_dummies(df['Sex'],drop_first=True)


# In[61]:


sns.countplot(data=df,x='Sex',color='skyblue',hue=df['Survived'])


# In[62]:


sns.pairplot(df)


# In[63]:


df['Age'].hist()


# In[64]:


sns.displot(x='Age',data=df)


# In[66]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[67]:


df['Survived'].value_counts()


# In[68]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,palette='RdBu_r')


# In[69]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')


# In[70]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')


# In[71]:


sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[72]:


df['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[73]:


sns.countplot(x='SibSp',data=df)


# In[74]:


df['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ### Data Cleaning 

# In[15]:


df=pd.read_csv('titanic_train.csv')


# In[16]:


df=df.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis=1)


# In[17]:


df=df.dropna()


# In[18]:


sns.heatmap(df.isnull(),cbar=False,yticklabels=False)


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def lr_process(): 
    X=df.drop('Survived',axis=1)
    y=df['Survived']
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3,random_state=101)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred=lr.predict(X_test)

    print(classification_report(y_test,y_pred))


# In[28]:


lr_process()


# # Dropped all categorical and null values

# In[29]:


df=pd.read_csv('titanic_train.csv')


# In[30]:


df=df.drop(['Name','Ticket','Cabin'],axis=1)


# In[31]:


df=df.dropna()


# In[35]:


df['Male']=pd.get_dummies(df['Sex'],drop_first=True)


# In[37]:


df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)


# In[39]:


df.drop(['Sex','Embarked'],axis=1,inplace=True)


# #### Dropped null values & coverted categorical into numerical values (Gender & embarked)

# In[40]:


lr_process()


# In[ ]:


df=pd.read_csv('titanic_train.csv')
df=df.drop(['Name','Ticket','Cabin'],axis=1)
df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)
df['Male']=pd.get_dummies(df['Sex'],drop_first=True)

df.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[41]:


for i in df.index:
    if (str(df['Age'][i]))=='nan':
        df.at[i,'Age']=np.average(df['Age'].dropna())


# In[42]:


sns.heatmap(df.isnull(),cbar=False,yticklabels=False)


# #### Filled Age Null Values with Average Age

# In[44]:


lr_process()


# #### Filled Age Null Values with Median Age

# In[45]:


df=pd.read_csv('titanic_train.csv')

df=df.drop(['Name','Ticket','Cabin'],axis=1)

df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)

df['Male']=pd.get_dummies(df['Sex'],drop_first=True)


df.drop(['Sex','Embarked'],axis=1,inplace=True)

for i in df.index:
    if (str(df['Age'][i]))=='nan':
        df.at[i,'Age']=np.median(df['Age'].dropna())

lr_process()


# #### Filled Age Null Values with Mode Age
# 

# In[46]:


df=pd.read_csv('titanic_train.csv')

df=df.drop(['Name','Ticket','Cabin'],axis=1)

df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)

df['Male']=pd.get_dummies(df['Sex'],drop_first=True)


df.drop(['Sex','Embarked'],axis=1,inplace=True)

for i in df.index:
    if (str(df['Age'][i]))=='nan':
        df.at[i,'Age']=24

lr_process()


# In[47]:


df=pd.read_csv('titanic_train.csv')

df=df.drop(['Name','Ticket','Cabin'],axis=1)

df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)

df['Male']=pd.get_dummies(df['Sex'],drop_first=True)


df.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[48]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[49]:


df['Age']=df[['Age','Pclass']].apply(impute_age,axis=1)


# #### Null Age Filled According to Pclass

# In[50]:


lr_process()


# In[51]:


df=pd.read_csv('titanic_train.csv')

df=df.drop(['Cabin'],axis=1)


df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)

df['Male']=pd.get_dummies(df['Sex'],drop_first=True)

df['Name']=(df['Name'].str.len())
df['Ticket']=(df['Ticket'].str.len())

df.drop(['Sex','Embarked'],axis=1,inplace=True)
for i in df.index:
    if (str(df['Age'][i]))=='nan':
        df.at[i,'Age']=24

lr_process()


# In[53]:


X


# In[ ]:




