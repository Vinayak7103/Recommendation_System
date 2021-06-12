#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT: 10

# ### TASK: RECOMMENDATION ENGINE

# To build a recommender system by using cosine simillarties score.

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


data=pd.read_csv('C:/Users/vinay/Downloads/book.csv',encoding='latin1')


# In[8]:


data.head()


# In[9]:


data=data.drop(['Unnamed: 0'],axis=1)


# In[10]:


data=data.rename(columns={'User.ID':'ID','Book.Title':'Book_Title','Book.Rating':'Book_Rating'})
data


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data.isnull().sum()


# In[14]:


data.corr()


# In[15]:


df=data.copy()


# In[16]:


df.head(2)


# In[17]:


df.ID.unique()


# In[13]:


n_users = df.ID.unique().shape[0]
n_users


# In[18]:


B_items = df.Book_Title.unique().shape[0]
B_items


# In[19]:


book_df = df.pivot_table(index='ID',
                   columns='Book_Title',
                   values='Book_Rating').reset_index(drop=True)


# In[20]:


book_df


# In[21]:


book_df.fillna(0, inplace=True)
book_df


# # One of the most basic metrics you can think of is the ranking to decide which top 250 books are based on their respective ratings.

# AVERAGE RATING OF BOOKS

# In[22]:


AVG = df['Book_Rating'].mean()
print(AVG)


# Next, let's calculate the number of ratings  received by a book in the 90th percentile. The pandas library makes this task extremely trivial using the .quantile() method of pandas:

# In[23]:


# Calculate the minimum number of votes required to be in the chart, 
minimum = data['Book_Rating'].quantile(0.90)
print(minimum)


# In[24]:


# Filter out all qualified Books into a new DataFrame
q_Books = data.copy().loc[data['Book_Rating'] >= minimum]
q_Books.shape


# In[25]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[26]:


uc_sim = 1 - pairwise_distances( book_df.values,metric='cosine')
uc_sim.shape


# In[27]:


uc_sim[1]


# In[28]:


#Store the results in a dataframe
uc_sim_df = pd.DataFrame(uc_sim)
uc_sim_df.index = df.ID.unique()
uc_sim_df.columns = df.ID.unique()


# In[29]:


uc_sim_df.iloc[0:10, 0:10]


# In[30]:


uc_sim_df.idxmax(axis=1)[0:125]


# In[31]:


users=df[(df['ID']==276726) | (df['ID']==17)]
users


# In[32]:


user_1=df[(df['ID']==276726)] 
user_1


# In[33]:


user_2=df[(df['ID']==17)] 
user_2


# In[34]:


indices = pd.Series(df.index, index=df['Book_Title']).drop_duplicates()


# In[35]:


indices[:10]


# In[36]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(Book_Title, cosine_sim=uc_sim):
    # Get the index of the books that matches the title
    idx = indices[Book_Title]

    # Get the pairwsie similarity scores of all books with that books
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[0:11]

    # Get the book indices
    books_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return df['Book_Title'].iloc[books_indices]


# In[37]:


a=get_recommendations('Under the Black Flag: The Romance and the Reality of Life Among the Pirates')
a


# In[38]:


b=get_recommendations('Classical Mythology')
b


# In[39]:


pd.merge(user_1,user_2,on='Book_Rating',how='outer')


# In[40]:


book_user_rating = book_df["You Don't Need Experience if You've Got Attitude"]  


# In[41]:


book_user_rating


# In[42]:


#Finding the correlation with different movies
similar_to_book = book_df.corrwith(book_user_rating) 


# In[44]:


corr_book = pd.DataFrame(similar_to_book, columns=['Correlation'])
corr_book.dropna(inplace=True)
corr_book.head()


# In[45]:


corr_book[corr_book['Correlation'] > 0].sort_values(by='Correlation', ascending=False).head(10)  


# In[46]:


ratings_mean_count = pd.DataFrame(df.groupby('Book_Title')['Book_Rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(df.groupby('Book_Title')['Book_Rating'].count())


# In[47]:


plt.style.use('dark_background')


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=10)


# In[49]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Book_Rating'].hist(bins=10)


# In[50]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Book_Rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)


#  CONCLUSION:-
#  
# From the output you can see that the Books that have high correlation with " are not very well known.
# 
# This shows that correlation alone is not a good metric for similarity because there can be a user who wished to take those Books  and only  other books and rated  them same.

# In[ ]:




