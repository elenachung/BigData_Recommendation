#!/usr/bin/env python
# coding: utf-8

# In[2]:


#data understanding
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff


# In[3]:


df = pd.read_csv('googleplaystore.csv', header=0, index_col=0, engine='python')


# In[4]:


df.describe()


# In[5]:


#describe the dataset 
print("types of each columns: \n\n",df.dtypes)
print("\ninfo about the columns: \n")
print(df.info())


# In[6]:


#data cleaning
df['Rating'] = df['Rating'].fillna(df['Rating'].median())
#df['Rating'].head(25)


# In[7]:


df.columns = df.columns.str.replace(' ','_') #replace column heads with _ instead of having spaces


# In[8]:


df.isnull().sum() #list the remaining nulls in categories


# In[9]:


#fill categorical with mode 
df['Type'].fillna(str(df.Type.mode().values[0]), inplace=True)
df['Current_Ver'].fillna(str(df.Current_Ver.mode().values[0]), inplace=True)
df['Android_Ver'].fillna(str(df.Android_Ver.mode().values[0]), inplace=True)

print(df.isnull().sum())


# In[10]:


#removing irrelevant data with ratings higher than 5
df.drop(df.loc[df['Rating'] > 5.0].index, inplace=True) 


# In[11]:


#no more NaN values
df.isnull().sum() 


# In[12]:


#check out Dtype and if nulls are presented for each column 
df.info()
df.sample(4)


# In[13]:


df['Reviews'] = df['Reviews'].astype(int) #object to integer

df['Size'] = np.where(df['Size'].str.startswith('Varies with device'), np.nan, df['Size']) #replace 'varies with device' with NaN
df['Size'] = df['Size'].str.rstrip('M') #strip 'M' - million in 'Size' col

df['Installs'] = df['Installs'].astype(str) #convert str to float and replace misc symbols (, and +)
df['Installs'] = df['Installs'].str.rstrip('+').replace(',','', regex=True)
df['Installs'] = df['Installs'].astype(float)

df['Price'] = df['Price'].astype(str) #remove '$' and convert to float
df['Price'] = df['Price'].str.lstrip('$')
df['Price'] = df['Price'].astype(float)

df.drop(columns = ['Current_Ver','Android_Ver']) #irrelevant data


# In[14]:


df.info()
df.sample(4)


# In[15]:


df['Genres'] = df['Genres'].str.replace('_&_',' ')


# In[17]:


plt.figure(figsize=(16,8))
p = sns.heatmap(df.corr(), annot=True, cmap="coolwarm")


# In[18]:


df['Category'].value_counts().sort_values(ascending=True) #sort the groups by number


# In[19]:


top_category_installs = df.groupby('Category')[['Installs']].sum().sort_values('Installs', ascending=False).head(15)
top_category_installs


# In[20]:


#top 10 ccategories 
bar = go.Bar(x=df.Category.value_counts().head(10).sort_values(ascending=True),
             y=df.Category.value_counts().head(10).sort_values(ascending=True).index,
             hoverinfo = 'x',
             text=df.Category.value_counts().head(10).sort_values(ascending=True).index,
             textposition = 'inside',
             orientation = 'h',
             opacity=0.6, 
             marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='Top 10 popular App category',
                   xaxis=dict(title="Count of reviews",),
                   margin = dict(l = 220),
                   font=dict(family='Arial',
                            color='black'))

fig = go.Figure(data=bar, layout=layout)

# Plot it
plotly.offline.iplot(fig)


# In[22]:


rate_app = df[['Category','Rating']].groupby('Category').agg('mean')

rate_app = rate_app.reset_index()

rate_app
bar4 = go.Bar(x=rate_app.sort_values(by="Rating",ascending=False).head(10).sort_values(by="Rating",ascending=True).Rating,
              y=rate_app.sort_values(by="Rating",ascending=False).head(10).sort_values(by="Rating",ascending=True).Category,
              hoverinfo = 'x',
              text=rate_app.sort_values(by="Rating",ascending=False).head(10).sort_values(by="Rating",ascending=True).Rating,
              textposition = 'inside',
              orientation = 'h',
              opacity=0.6, 
              marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='Top 10 Apps with Highest Ratings',
                   xaxis=dict(title="Average Ratings",), margin = dict(l = 220),
                   font=dict(family='Arial',
                            color='black'))

fig = go.Figure(data=bar4, layout=layout)

plotly.offline.iplot(fig)


# In[25]:


plt.figure(figsize=(30,10),facecolor= 'white',edgecolor='black')
df['Category'].value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.show()


# In[42]:


def app_types(df):
    """
    Computing free and paid apps 
    """
    return sum(df.Type == "Free"), sum(df.Type == 'Paid')


# In[44]:


def plot_app_types(df):
    """
    Plot app type distributions across categories
    """
    rating = df.Category.value_counts()
    cat_free = []
    cat_paid = []
    for cat in rating.index:
        n_free, n_paid = app_types(df.query("Category == '{}'".format(cat)))
        cat_free_apps.append(n_free)
        cat_paid_apps.append(n_paid)

    f, ax = plt.subplots(2,1)
    ax[0].bar(range(1, len(cat_free)+1), cat_free)
    ax[1].bar(range(1, len(cat_free)+1), cat_paid)


# In[47]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))
plt.suptitle('Count plots')
sns.countplot(y='Category',data=df,ax=ax1)
sns.countplot('Type',data=df,ax=ax2)
plt.show()


# In[84]:


df.to_csv('Cleaned_google_play_store.csv')


# In[ ]:




