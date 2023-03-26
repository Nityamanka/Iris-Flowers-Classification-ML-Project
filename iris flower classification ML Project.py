#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[22]:


from sklearn.datasets import load_iris


# In[23]:


dataset = load_iris()


# In[4]:


dataset


# In[5]:


print(dataset.DESCR)


# In[6]:


x = dataset.data


# In[7]:


y = dataset.target


# In[8]:


y


# In[9]:


x


# In[24]:


x.shape


# In[25]:


y.shape


# In[26]:


plt.plot(x[:,0][y==0]*x[:,1][y==0],x[:,1][y==0]*x[:,2][y==0],'r.',label='Setosa')
plt.plot(x[:,0][y==1]*x[:,1][y==1],x[:,1][y==1]*x[:,2][y==1],'g.',label='Versicolour')
plt.plot(x[:,0][y==2]*x[:,1][y==2],x[:,1][y==2]*x[:,2][y==2],'b.',label='Virginica')
plt.legend()
plt.show()



# In[12]:


from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train, x_test , y_train , y_test = train_test_split(x,y)


# In[27]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)


# In[32]:


log_reg.score(x,y)


# In[31]:


log_reg.score(x_train,y_train)


# In[30]:


log_reg.score(x_test,y_test)


# In[ ]:




