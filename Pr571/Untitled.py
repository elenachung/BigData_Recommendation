#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


sns.set()


# In[3]:


googs_df = pd.read_csv("googleplaystore.csv",encoding='utf-8', header=0, engine='python')


# In[4]:


googs_df


# In[5]:


googs_df.isna().sum()


# In[6]:


googs_df[googs_df['Reviews']=='3.0M']


# In[7]:


for i in range(len(googs_df.columns)-1,1,-1):
    googs_df.loc[[googs_df[googs_df['Category']=='1.9'].index[0]],[googs_df.columns[i]]]=googs_df.loc[[googs_df[googs_df['Category']=='1.9'].index[0]],[googs_df.columns[i-1]]].values
googs_df.loc[[googs_df[googs_df['Category']=='1.9'].index[0]],['Category']]='ART_AND_DESIGN'


# In[8]:


googs_df['Reviews']=googs_df['Reviews'].astype(int)


# In[9]:


googs_df['Price'].replace(to_replace='0',value='$0',inplace=True)
googs_df['Price']=googs_df['Price'].apply(lambda a : a[1:])
googs_df['Price']=googs_df['Price'].astype(float)


# In[10]:


googs_df['Size'].replace(to_replace='Varies with device',value='0M',inplace=True)
googs_df['Size']=googs_df['Size'].apply(lambda a : a.replace(',',''))


# In[11]:


googs_df['Value']=googs_df['Size'].apply(lambda a : a[:-1])
googs_df['Unit']=googs_df['Size'].apply(lambda a : a[-1:])
googs_df['Value']=googs_df['Value'].astype(float)


# In[12]:


googs_df['Installs'].replace(to_replace='0',value='0+',inplace=True)
googs_df['Installs']=googs_df['Installs'].apply(lambda a : a.replace(',',''))

googs_df['Installs']=googs_df['Installs'].apply(lambda a : a[:-1])
googs_df['Installs']=googs_df['Installs'].astype(int)


# In[13]:


googs_df['Rating']=googs_df['Rating'].astype('float')


# In[14]:


# googs_df[googs_df['Genres'].isnull()]


# In[15]:


googs_df.loc[[googs_df[googs_df['Genres'].isnull()].index[0]],['Genres']]='Art & Design'


# In[16]:


# googs_df[googs_df['Type'].isnull()]


# In[17]:


googs_df['Type'].fillna(googs_df['Type'].mode()[0],inplace=True)


# In[18]:


# googs_df[googs_df['Android Ver'].isnull()]


# In[19]:


googs_df['Android Ver'].fillna(googs_df['Android Ver'].mode()[0],inplace=True)


# In[20]:


googs_df[googs_df['Current Ver'].isnull()]


# In[21]:


temp=pd.DataFrame()
for i in googs_df['Category'].unique():
    temp1=googs_df[(googs_df['Category']==i)]['Rating'].fillna(googs_df[(googs_df['Category']==i)]['Rating'].mode()[0])
    temp=pd.concat([temp,temp1])


# In[22]:


googs_df['Rating']=temp


# In[23]:


googs_df.info()


# ### Recommendation System

# In[24]:


app_recom=googs_df[(googs_df['Installs']>1000) & (googs_df['Reviews']>100)][['Category', 'Reviews','Rating', 'Installs','Price', 'Content Rating','Genres']]


# In[25]:


app_recom=pd.get_dummies(app_recom,columns=['Category','Content Rating','Genres'],prefix='',prefix_sep='')


# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


sc=StandardScaler()
sc.fit(app_recom)
app_recom=pd.DataFrame(sc.transform(app_recom))


# In[28]:


app_recom.index=googs_df[(googs_df['Installs']>1000) & (googs_df['Reviews']>100)]['App']


# In[29]:


app_recom=app_recom.T
app_recom.head()


# In[30]:


for_app=app_recom['Facebook']


# In[31]:


recommend=pd.DataFrame(app_recom.corrwith(for_app),columns=['Correlation'])
recommend.sort_values('Correlation',ascending=False).head(10)


# In[ ]:




