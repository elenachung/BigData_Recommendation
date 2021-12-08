#!/usr/bin/env python
# coding: utf-8

# In[236]:


#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import itertools
from numpy import linalg
import matplotlib as mpl
from sklearn import mixture


# In[237]:
    
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


#load dataset as dataframe 
df = pd.read_csv(r'Cleaned_google_play_store.csv', header=0, index_col=0, engine='python')


# In[284]:


#shape of df
print(df.head(2))
print(df.shape)


# In[239]:


#preprocessing steps 
df.info()


# In[240]:


#drop irrelevant columns
df.drop(['Size','Genres', 'Last_Updated', 'Current_Ver', 'Android_Ver'], axis=1, inplace=True)


# In[242]:


#convert df to string and float
df[['Category', 'Type', 'Content_Rating']] = df[['Category', 'Type', 'Content_Rating']].astype(str)
df['Rating'] = df[['Rating']].astype(float)


# In[243]:


#view dtypes
df.dtypes


# In[288]:


#one hot encoder 
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)


# In[245]:


#one hot encoder for multiple features
features_to_encode = ['Category', 'Type', 'Content_Rating']
for feature in features_to_encode:
    df = encode_and_bind(df, feature)


# In[269]:


#list all columns so its easier to view columnn names
dfcol = df.columns.values.tolist()
lst = [dfcol[x:x+1] for x in range(0, len(dfcol),1)] 
len(lst)


# In[270]:


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


# In[280]:


#create array using points
X = np.array(list(zip(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,
                     f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,
                     f38,f39,f40,f41,f42,f43,f44,f45)))


# In[281]:


#k-means clustering algorithm with various paratmeters
kmeans = KMeans(init="random", n_clusters=3, n_init=5, max_iter=300, random_state=42)
kmeans.fit(X)


# In[282]:


#which labels are clusters
kmeans.cluster_centers_


# In[289]:


#4 clusters with red circles
kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=500, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()


# In[287]:


print(pred_y)


# In[ ]:
gmm = mixture.GaussianMixture(n_components=5).fit(X)

pr = [2.0, 3.0, 4.0]
tr = {}
thrs = ['low', 'med', 'high']
kel = gmm.predict(X)
for i in range(3):
    tr[thrs[i]] = kel[kel >= pr[i]].mean()

plt.scatter(x=tr.keys(), y=tr.values())
plt.yscale("linear")