#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("C:\TALENT BATTLE\Machine Learning Project\Week3\spam.csv",encoding="latin-1")


# In[52]:


df.head(n=15)


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[8]:


import re
import nltk
from nltk.corpus import stopwords
import string


# In[11]:


nltk.download('stopwords')


# In[12]:


stemmer=nltk.SnowballStemmer("english")


# In[13]:


stopword=set(stopwords.words('english'))


# In[14]:


def clean(text):
    text = str(text) . lower()
    text = re. sub('\[.*?\]',' ',text)
    text = re. sub('https?://\S+/www\. \S+', ' ', text)
    text = re. sub('<. *?>+', ' ', text)
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)
    text = [word for word in text. split(' ') if word not in stopword]
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]
    text = " ". join(text)
    return text


# In[15]:


df['message']=df['message'].apply(clean)


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator


# In[21]:


text=" ".join(i for i in df. message)


# In[22]:


stopwords=set(STOPWORDS)


# In[23]:


wordcloud=WordCloud(stopwords=stopwords,background_color="white").generate(text)


# In[24]:


plt.figure(figsize=(10,10))


# In[25]:


plt.imshow(wordcloud)


# In[31]:


# plt.axis("off")


# In[32]:


# plt.show()


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x=np.array(df["message"])
y=np.array(df["class"])


# In[36]:


cv=CountVectorizer()
X=cv.fit_transform(x)


# In[38]:


print(X)


# In[39]:


xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33)


# In[40]:


from sklearn.naive_bayes import BernoulliNB


# In[41]:


model=BernoulliNB()


# In[42]:


model.fit(xtrain,ytrain)


# In[56]:


user=input("Enter the text")
data=cv.transform([user]).toarray()
print(model.predict(data))


# In[ ]:




