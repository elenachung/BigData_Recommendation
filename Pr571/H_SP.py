#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


# In[3]:


#load dataset as dataframe 
df = pd.read_csv('Downloads/Cleaned_google_play_store.csv', header=0, index_col=0, engine='python')


# In[ ]:


#shape of df
print(df.head(2))
print(df.shape)


# In[108]:


#preprocessing steps 
df.info()


# In[109]:


#drop irrelevant columns
df.drop(['Size','Genres', 'Last_Updated', 'Current_Ver', 'Android_Ver'], axis=1, inplace=True)


# In[110]:


#convert df to string and float
df[['Category', 'Type', 'Content_Rating']] = df[['Category', 'Type', 'Content_Rating']].astype(str)
df['Rating'] = df[['Rating']].astype(float)


# In[111]:


#view dtypes
df.dtypes


# In[112]:


#one hot encoder 
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)


# In[113]:


#one hot encoder for multiple features
features_to_encode = ['Category', 'Type', 'Content_Rating']
for feature in features_to_encode:
    df = encode_and_bind(df, feature)


# In[114]:


#list all columns so its easier to view columnn names
dfcol = df.columns.values.tolist()
lst = [dfcol[x:x+1] for x in range(0, len(dfcol),1)] 
len(lst)


# In[102]:


#extract each column - find easier way
f1 = df['Rating']
f2 = df['Reviews']
f3 = df['Installs']
f4 = df['Price']
f5 = df['Category_ART_AND_DESIGN']
f6 = df['Category_AUTO_AND_VEHICLES']
f7 = df['Category_BEAUTY']
f8 = df['Category_BOOKS_AND_REFERENCE']
f9 = df['Category_BUSINESS']
f10 = df['Category_COMICS']
f11 = df['Category_COMMUNICATION']
f12 = df['Category_DATING']
f13 = df['Category_EDUCATION']
f14 = df['Category_ENTERTAINMENT']
f15 = df['Category_EVENTS']
f16 = df['Category_FAMILY']
f17 = df['Category_FINANCE']
f18 = df['Category_FOOD_AND_DRINK']
f19 = df['Category_GAME']
f20 = df['Category_HEALTH_AND_FITNESS']
f21 = df['Category_HOUSE_AND_HOME']
f22 = df['Category_LIBRARIES_AND_DEMO']
f23 = df['Category_LIFESTYLE']
f24 = df['Category_MAPS_AND_NAVIGATION']
f25 = df['Category_MEDICAL']
f26 = df['Category_NEWS_AND_MAGAZINES']
f27 = df['Category_PARENTING']
f28 = df['Category_PERSONALIZATION']
f29 = df['Category_PHOTOGRAPHY']
f30 = df['Category_PRODUCTIVITY']
f31 = df['Category_SHOPPING']
f32 = df['Category_SOCIAL']
f33 = df['Category_SPORTS']
f34 = df['Category_TOOLS']
f35 = df['Category_TRAVEL_AND_LOCAL']
f36 = df['Category_VIDEO_PLAYERS']
f37 = df['Category_WEATHER']
f38 = df['Type_Free']
f39 = df['Type_Paid']
f40 = df['Content_Rating_Adults only 18+']
f41 = df['Content_Rating_Everyone']
f42 = df['Content_Rating_Everyone 10+']
f43 = df['Content_Rating_Mature 17+']
f44 = df['Content_Rating_Teen']
f45 = df['Content_Rating_Unrated']


# In[115]:


#create array using points
X = np.array(list(zip(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,
                     f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,
                     f38,f39,f40,f41,f42,f43,f44,f45)))


# In[116]:


#k-means clustering algorithm with various paratmeters
kmeans = KMeans(init="random", n_clusters=3, n_init=5, max_iter=300, random_state=42)
kmeans.fit(X)


# In[117]:


#which labels are clusters
kmeans.cluster_centers_


# In[118]:


#4 clusters with red circles
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=500, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()


# In[119]:


print(pred_y)


# Clustering 
# It is the process of automatically grouping data points together that has similar characteristics and assigning to clusters. This is really helpful for recommender systems which groups user with similar viewing patterns or downloads on certain app category in order to recommend something similar. This is based on the similarity of how close two items are that will results in recommending something similar. It calculates the vector distances between these items and depending on those distances we will know how similar they are and it will be easier to recommend something similar to that app. 

# In[120]:


app_names = pd.DataFrame({'App':df.index})


# In[122]:


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
model = NearestNeighbors(metric = 'cosine')
model.fit(df)


# In[123]:


def getSimilarApps(appname, recommend_apps=100, get_similarity = False):
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


# In[89]:


app_names.loc[[300]]


# In[124]:


getSimilarApps(app_names.loc[300], get_similarity = True)


# In[ ]:




