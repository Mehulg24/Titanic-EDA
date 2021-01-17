#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as dt


# In[86]:


path = "C:\\Users\Mehul Gupta\\OneDrive\\Documents\\GitHub\\EDA1"


# In[87]:


os.chdir(path)
os.listdir()


# In[88]:


train = pd.read_csv("titanic_train.csv")
train.head()


# # Checking missing value 

# In[89]:


train.isnull().sum()


# In[90]:


sns.heatmap(data = train.isnull(), yticklabels=False, cmap=('viridis'))


# In[91]:


sns.countplot(data = train, x = "Survived")


# In[92]:


sns.countplot(data = train, x = 'Survived', hue = 'Sex',palette="rainbow" )


# In[93]:


sns.countplot(data = train, hue = 'Pclass', x = 'Survived', palette="rainbow")


# In[94]:


sns.distplot(train['Age'].dropna(),bins=40, kde = False, color= 'red' )


# In[95]:


sns.distplot(train['Age'], )


# In[96]:


sns.countplot(x = "SibSp", data = train)


# In[97]:


train['Fare'].hist(bins=40)


# In[98]:


plt.figure(figsize= (12,8))
sns.boxplot(data = train, x = 'Pclass', y = 'Age')


# In[99]:


def Avg_Age(Cols):
    Age = Cols[0]
    Pclass = Cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else :
            return 24 
    else :
        return Age
    


# In[100]:


train['Age'] = train[['Age', 'Pclass']].apply(Avg_Age, axis = 1)


# In[101]:


sns.heatmap(data = train.isnull(), yticklabels=False, cmap=('viridis'))


# In[102]:


train.drop('Cabin', inplace=True , axis = 1)


# In[103]:


train


# In[104]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)
sex =  pd.get_dummies(train['Sex'], drop_first=True)


# In[105]:


train.drop(['Sex','Embarked', 'Ticket', 'Name' ], axis = 1, inplace=True )


# In[106]:


train.head()


# In[107]:


train = pd.concat([train, sex, embark], axis = 1)


# In[108]:


train


# In[110]:


y = train.pop('Survived')


# In[ ]:


y.head()


# In[113]:


X = train


# In[114]:


X_train,X_test, y_train, y_test = train_test_split(
                                                    X,    # Predictors
                                                    y,                   # Target
                                                    test_size = 0.3      # split-ratio
                                                    )


# # This is First example to determine accuracy via Logistics Regression

# In[116]:


clf_LR = LogisticRegression()


# In[117]:


clf_LR.fit(X_train,y_train)


# In[118]:


clf_pred = clf_LR.predict(X_test)


# In[120]:


np.sum(clf_pred == y_test)/y_test.values.size
clf_pred


# In[122]:


np.sum(clf_pred == y_test)/y_test.values.size


# # This is Second example to determine accuracy via Logistics Regression

# In[123]:


from sklearn.metrics import confusion_matrix


# In[124]:


accuracy = confusion_matrix(y_test, clf_pred)
accuracy 


# In[127]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,clf_pred)
accuracy


# # This exaample to determine accuracy via Decision Tree

# In[130]:


clf_DT = dt()


# In[131]:


clf_DT.fit(X_train, y_train)


# In[132]:


clf_dt_pred = clf_DT.predict(X_test)


# In[133]:


clf_dt_pred


# In[134]:


np.sum(clf_dt_pred == y_test)/y_test.values.size


# by comparing both algorithem we found that logistics Regression giving best accuracy 

# In[ ]:




