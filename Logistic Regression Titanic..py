
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### EDA

df=pd.read_csv('titanic_train.csv')


df.head()

sns.countplot(data=df,x='Survived',hue='Sex')


df['Survived'].value_counts().plot(kind='bar')


pd.get_dummies(df['Sex'],drop_first=True)


sns.countplot(data=df,x='Sex',color='skyblue',hue=df['Survived'])

sns.pairplot(df)


df['Age'].hist()


sns.displot(x='Age',data=df)


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


df['Survived'].value_counts()


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,palette='RdBu_r')


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')

sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=30)


df['Age'].hist(bins=30,color='darkred',alpha=0.7)


sns.countplot(x='SibSp',data=df)

df['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ### Data Cleaning 


df=pd.read_csv('titanic_train.csv')


df=df.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis=1)

df=df.dropna()

sns.heatmap(df.isnull(),cbar=False,yticklabels=False)

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


lr_process()


# # Dropped all categorical and null values


df=pd.read_csv('titanic_train.csv')


df=df.drop(['Name','Ticket','Cabin'],axis=1)


df=df.dropna()


df['Male']=pd.get_dummies(df['Sex'],drop_first=True)

df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)

df.drop(['Sex','Embarked'],axis=1,inplace=True)


# #### Dropped null values & coverted categorical into numerical values (Gender & embarked)


lr_process()


df=pd.read_csv('titanic_train.csv')
df=df.drop(['Name','Ticket','Cabin'],axis=1)
df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)
df['Male']=pd.get_dummies(df['Sex'],drop_first=True)

df.drop(['Sex','Embarked'],axis=1,inplace=True)

for i in df.index:
    if (str(df['Age'][i]))=='nan':
        df.at[i,'Age']=np.average(df['Age'].dropna())


sns.heatmap(df.isnull(),cbar=False,yticklabels=False)


# #### Filled Age Null Values with Average Age


lr_process()


# #### Filled Age Null Values with Median Age


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


df=pd.read_csv('titanic_train.csv')

df=df.drop(['Name','Ticket','Cabin'],axis=1)

df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)

df['Male']=pd.get_dummies(df['Sex'],drop_first=True)


df.drop(['Sex','Embarked'],axis=1,inplace=True)

for i in df.index:
    if (str(df['Age'][i]))=='nan':
        df.at[i,'Age']=24

lr_process()


df=pd.read_csv('titanic_train.csv')

df=df.drop(['Name','Ticket','Cabin'],axis=1)

df[['Q','S']]=pd.get_dummies(df['Embarked'],drop_first=True)

df['Male']=pd.get_dummies(df['Sex'],drop_first=True)


df.drop(['Sex','Embarked'],axis=1,inplace=True)


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


df['Age']=df[['Age','Pclass']].apply(impute_age,axis=1)


# #### Null Age Filled According to Pclass


lr_process()


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





