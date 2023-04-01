import numpy as py 
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('spam_ham_dataset.csv')

X = df['text']
y = df['label'] 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)

mdl = MultinomialNB()
mdl.fit(X_train_count,y_train)
print(mdl.score(X_train_count,y_train)*100)

