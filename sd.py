import pandas as pd
import numpy as np
import pickle

df = pd.read_excel('spam.xlsx', header=None, names=['class', 'sms'])
df['label'] = df['class'].map({'ham':0, 'spam':1})

x = df['sms']
y = df['label']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 1)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english')

cv.fit(x_train)

xtrt = cv.transform(x_train)
xtst = cv.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
m = MultinomialNB()

m.fit(xtrt, y_train)

pickle.dump(m, open('mdl.pkl','wb'))
pickle.dump(cv.vocabulary_, open('vocabulary_to_load.pkl','wb'))
