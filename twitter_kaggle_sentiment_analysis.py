# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 23:49:30 2019

@author: HIMANSHU
"""
#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Importing the dataset
dataset = pd.read_csv('Tweets.csv')

dataset.info()

'''#accessing values $not allinging with the below code
airline_values = list(dataset['airline_sentiment'].value_counts().unique())'''

#accessing name of airline
airline_name = list(dataset['airline'].unique())

'''df = dataset[dataset['airline']=='Virgin America'] #[] is working while () is not working
y = df['airline_sentiment'].value_counts() #index are arranged alphabetically
y.index #this work, while y.index()is not working'''
#plotting review
for i in range(6):
    plt.subplot(3,2,i+1)
    df = dataset[dataset['airline']==airline_name[i]]
    y = df['airline_sentiment'].value_counts()
    x = y.index
    plt.bar(x,y)
    plt.ylabel('Count')
    plt.xlabel('Sentiment Type')
    plt.title('Count of Sentiment of '+airline_name[i])
plt.subplots_adjust(wspace=0.6,hspace=1.4)
plt.show()

#plotting negative reasons
for i in range(6):
    df = dataset[dataset['airline']==airline_name[i]]
    y = df['negativereason'].value_counts()
    x = y.index
    plt.bar(x,y)
    plt.ylabel('Reason Count')
    plt.xlabel('Reason Name')
    plt.xticks(x,rotation=90)
    plt.title('Count of Moods of '+airline_name[i])
    plt.show()

from wordcloud import WordCloud
#collect words
#first lower then split bcoz list obj. has no 'lower' naem attribute
corpus = []
text_series = dataset['text']
for i in text_series.index:
    text = text_series[i].lower()
    text = text.split()    
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    text = re.sub('@[a-zA-Z]+','', text)
    corpus.append(text)
    
all_words = ' '.join([word for word in corpus])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

corpus1 = []
text_series1 = dataset[dataset['airline_sentiment']=='negative']['text']
for i in text_series1.index:
    text = text_series1[i].lower()
    text = text.split()  
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    text = re.sub('@[a-zA-Z]+','', text)
    corpus1.append(text)
    
all_words1 = ' '.join([word for word in corpus1])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words1)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_test, y_pred)
f1_score = f1_score(y_test, y_pred, average='micro')

