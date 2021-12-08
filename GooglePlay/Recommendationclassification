#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


#load dataset as dataframe 
df = pd.read_csv('Cleaned_google_play_store.csv', header=0, index_col=0, engine='python')


# In[3]:


#shape of df
print(df.head(2))
print(df.shape)


# In[4]:


#preprocessing steps 
df.info()


# In[5]:


#drop irrelevant columns
df.drop(['Size','Genres', 'Last_Updated', 'Current_Ver', 'Android_Ver'], axis=1, inplace=True)


# In[6]:


#convert df to string and float
df[['Category', 'Type', 'Content_Rating']] = df[['Category', 'Type', 'Content_Rating']].astype(str)
df['Rating'] = df[['Rating']].astype(float)


# In[7]:


#view dtypes
df.dtypes


# In[8]:


#one hot encoder 
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)


# In[9]:


#one hot encoder for multiple features
features_to_encode = ['Category', 'Type', 'Content_Rating']
for feature in features_to_encode:
    df = encode_and_bind(df, feature)


# In[10]:


app_names = pd.DataFrame({'App':df.index})


# In[11]:


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
model = NearestNeighbors(metric = 'euclidean')
model.fit(df)


# In[12]:


def getSimilarApps(appname, recommend_apps=20, get_similarity = False):
    distances,neighbors = model.kneighbors(df.loc[appname], n_neighbors=recommend_apps+1)
    print(f'Apps like ' + ''+ str(appname[0]) + ''+ 'that you would like...')
    print(neighbors[0][1:])
    similar_apps =[]
    for neighbor in neighbors[0][1:]:
        similar_apps.append(app_names.loc[neighbor][0])
    if not get_similarity:
        return similar_apps
    similarity = []
    for app in similar_apps:
        sim = cosine_similarity(df.loc[[appname[0]]],df.loc[[app]]).flatten()[0]
        similarity.append(sim*100)
    sim_df = pd.DataFrame({'App':similar_apps, 'Similarity':similarity})
    sim_df.sort_values(by='Similarity', ascending =False)
    return sim_df


# In[13]:


app_names.loc[[300]]


# In[14]:


getSimilarApps(app_names.loc[1], get_similarity = True)


# In[15]:


app_names_1 = pd.DataFrame({'App':df.index})
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
model = NearestNeighbors(metric = 'cosine')
model.fit(df)
def getSimilarApps(appname, recommend_apps=20, get_similarity = False):
    distances,neighbors = model.kneighbors(df.loc[appname], n_neighbors=recommend_apps+1)
    print(f'Apps like ' + ''+ str(appname[0]) + ''+ 'that you would like...')
    print(neighbors[0][1:])
    similar_apps =[]
    for neighbor in neighbors[0][1:]:
        similar_apps.append(app_names_1.loc[neighbor][0])
    if not get_similarity:
        return similar_apps
    similarity = []
    for app in similar_apps:
        sim = cosine_similarity(df.loc[[appname[0]]],df.loc[[app]]).flatten()[0]
        similarity.append(sim*100)
    sim_df = pd.DataFrame({'App':similar_apps, 'Similarity':similarity})
    sim_df.sort_values(by='Similarity', ascending =False)
    return sim_df


# In[16]:


getSimilarApps(app_names_1.loc[1], get_similarity = True)


# In[ ]:




