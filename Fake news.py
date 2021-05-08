#!/usr/bin/env python
# coding: utf-8

# # A machine learning model has been made to identify when an article might be fake news

# ## Importing libraries which will be in use

# In[1]:


import nltk
import pandas as pd
import numpy as np
import tensorflow as tf


# In[4]:


# Reading training dataset
train = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\fake news\train.csv')


# In[5]:


# Top 5 rows
train.head()


# In[6]:


#total rows and columns
train.shape


# In[7]:


# Null values in the dataset
train.isnull().sum()


# In[8]:


# Dropping all the null values
train.dropna(inplace=True)


# In[9]:


# Again checking null values
train.isnull().sum()


# In[10]:


# Making a copy of train dataset and storing in message 
message = train.copy()


# In[11]:


# Resetting index because we have remove null values so the indexes are not alinged  in order. 
message.reset_index(inplace=True)


# In[12]:


message.head(15)


# In[13]:


message['title'][5]


# In[14]:


message.shape


# In[15]:


# Storing values of text column in col variable
col = message['text']


# ## Stemming on train data

# In[16]:


import re


# In[17]:


# importing nltk objects which will be in further use.
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps =  PorterStemmer()


# In[18]:


# using for loop to apply stemming for every word.
corpus=[]
for i in range(len(message)):
    review= re.sub('[^a-zA-Z]', ' ', col[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)


# In[19]:


corpus


# In[20]:


corpus[5]


# In[21]:


len(corpus)


# ## TF-IDF on train data
# 

# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()


# In[23]:


X.shape


# In[24]:


X


# In[25]:


len(tfidf.get_feature_names())


# In[26]:


# Storing values of output variables in y.
y = train['label'].values


# ## Applying Artificial Neural Network Model

# In[27]:


ann = tf.keras.models.Sequential()
# Adding neural network layers
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='softmax'))


# In[28]:


# Comppilation of ANN model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[29]:


# Fitting the ANN model
ann.fit(X, y, batch_size=32, epochs=100)


# ### Applying Tf-Idf vectorizor on our test dataset

# In[30]:


test = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\fake news\test.csv')


# In[31]:


test.head()


# In[32]:


test.isnull().sum()


# In[33]:


test.dropna(inplace=True)


# In[34]:


message_test = test.copy()


# In[35]:


message_test.reset_index(inplace=True)


# In[36]:


col2 = message_test['text']


# In[37]:


col2[6]


# In[38]:


corpus2=[]
for i in range(len(message_test)):
    reviewe= re.sub('[^a-zA-Z]', ' ', col2[i])
    reviewe = reviewe.lower()
    reviewe = reviewe.split()
    reviewe = [ps.stem(word) for word in reviewe if word not in set(stopwords.words('english'))]
    reviewe= ' '.join(reviewe)
    corpus2.append(reviewe)


# In[39]:


corpus2


# In[40]:


X_test = tfidf.transform(corpus2).toarray()


# In[41]:


X_test.shape


# In[42]:


X_test


# In[43]:


X_test.shape


# ## Predicting X_test Values

# In[44]:


predicted_values = ann.predict(X_test)


# In[45]:


predicted_values


# ## Storing predicted values into a CSV file 

# In[46]:


Dataframe = pd.DataFrame(predicted_values, columns=['label']) 


# In[47]:


Dataframe.to_csv (r'C:\Users\The ChainSmokers\Desktop\fakenews_Kaggle.csv', index = False, header=True)


# In[ ]:




