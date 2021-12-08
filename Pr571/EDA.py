#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


googs_df = pd.read_csv("googleplaystore.csv",encoding='utf-8')


# In[3]:


googs_df


# In[4]:


googs_df.columns


# In[5]:


googs_df.info()


# In[6]:


googs_df.isnull().sum()


# ### Data Cleaning

# In[7]:


data = googs_df[pd.notnull(googs_df['Rating'])]


# In[8]:


median_rating = np.median(data['Rating'])
googs_df['Rating'].fillna(median_rating)


# In[9]:


googs_df.dropna()
googs_df.isnull().sum()


# In[10]:


googs_df.drop_duplicates() 


# In[11]:


googs_df.info()


# ### Exploratory Data Analysis

# In[28]:


#  Most popular category
plt.figure(figsize=(30,10),facecolor= 'white',edgecolor='black')
googs_df['Category'].value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.show()


# In[13]:


#  Content Rating 
plt.figure(figsize=(30,10),facecolor= 'white',edgecolor='black') 
googs_df['Content Rating'].value_counts().plot(kind='pie',autopct="%i%%")
plt.show()

plt.figure(figsize=(10,5),facecolor= 'white',edgecolor='black')
googs_df['Content Rating'].value_counts().plot(kind='bar',color = 'green')
plt.xlabel('Content Rating')
plt.ylabel('count')
plt.show()


# In[14]:


### free vs paid

plt.figure(figsize=(30,10),facecolor= 'white',edgecolor='black')
googs_df['Type'].value_counts().plot(kind='pie',autopct="%i%%")
plt.legend()
plt.show()
plt.figure(figsize=(10,5),facecolor= 'white',edgecolor='black')
data['Type'].value_counts().plot(kind='bar',color = 'green')
plt.xlabel('Type')
plt.ylabel('count')
plt.show()


# In[15]:


##App that has the largest number of reviews

googs_df['Reviews'] = pd.to_numeric(data['Reviews'],errors='coerce')
googs_df.sort_values('Reviews',ascending=False).iloc[0]['App']


# In[16]:


#largest size App

googs_df.sort_values('Size',ascending=False)
googs_df.loc[googs_df['Size'] == 'Varies with device'].shape

remove = googs_df.loc[googs_df['Size'] == 'Varies with device']

googs_df.drop(remove.index,inplace = True)

#Removing the unwanted data 
googs_df['Size'] = googs_df['Size'].apply(lambda x: str(x).replace('M',''))
googs_df['Size'] = googs_df['Size'].apply(lambda x: str(x).replace('k',''))
googs_df['Size'] = googs_df['Size'].apply(lambda x: str(x).replace('+',''))


googs_df['Size'] = pd.to_numeric(googs_df['Size'],errors='coerce')
googs_df.sort_values('Size', ascending=False).iloc[0]['App']


# In[17]:


### App that has the highest number of installs 
googs_df['Installs'] = googs_df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
googs_df['Installs'] = pd.to_numeric(googs_df['Installs'],errors='coerce')
googs_df.sort_values('Installs', ascending=False).iloc[0]['App']


# In[18]:


##App with most reviews
sort = googs_df.sort_values('Reviews',ascending = False )[:40]
plot = sns.barplot(x = 'Reviews' , y = 'App' , data = sort )
plot.set_xlabel('Reviews')
plot.set_ylabel('')
plot.set_title("Most Reviwed Apps in Play Store", size = 20)

