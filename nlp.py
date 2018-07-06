# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:18:53 2018

@author: Sybille Legitime
"""

#Natural Language Processing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #Keeps all letters in each review and separates them with a space
    review = review.lower() #puts all characters in lowercase
    review = review.split() #splitting the review from a string of characters to alist of its different words
    ps = PorterStemmer() #takes the root of the word (i.e 'loving', 'loved', and 'will love' will turn into 'love')
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #go through each word and add the words that are not stopwords to the list
    review = ' '.join(review) #transform the list back to a string, separated by spaces
    corpus.append(review) #add cleaned review to corpus
    
#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #sparse matrix of features
y = dataset.iloc[:, 1].values #dependent variable vector

'''Use Naive Bayes Classification to train the machine '''
#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Naive Bayes into the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
