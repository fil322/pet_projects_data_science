"""
This advanced python project of detecting fake news deals with fake and real news. 
Using sklearn, we build a TfidfVectorizer on our dataset. 
Then, we initialize a PassiveAggressive Classifier and fit the model. 
In the end, the accuracy score and the confusion matrix tell us how well our model fares.
"""

"""
pip install numpy pandas sklearn
"""


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#Read the data
df = pd.read_csv("D:\\code\\data_science\\Fake_news_pproject\\news.csv")

#Get shape and head
df.shape
df.head()

#DataFlair - Get the labels
labels=df.label
labels.head()

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#DataFlair - Build confusion matrix
cm = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(f'Negatives: True:{cm[1][1]} False:{cm[0][1]}')
print(f'Positives: True:{cm[0][0]} False:{cm[1][0]}')