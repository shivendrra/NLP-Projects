import numpy as np
import pandas as pd
import os 

# importing the file
path = os.chdir("d:/Machine Learning/NLP/CSV Files")
data = pd.read_csv("anthems.csv")

# separating the features
para = data['Anthem'].values.astype("U")  #containes the values of row 4
drow = data.iloc[:,4]  #containes row 1 and 4 only
sen_arr = np.array(drow)  #converts normal comma separated values into array

# preprocess the data
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
corpus = []
words = []
for i in range(len(para)):
    sentence = para[i]
    sent_token = nltk.sent_tokenize(sentence)
    for i in range(0, len(sent_token)):
        review = re.sub('[^a-zA-Z]', ' ', sent_token[i])
        review = review.lower()
        review = review.split()
    
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
    corpus.append(review)

# applying tfidf  
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x_feat = cv.fit_transform(corpus).toarray()

# K-means algorithm
from sklearn.cluster import KMeans
k = 5
model = KMeans(n_clusters=k, random_state=40)
kmeans = model.fit(x_feat)
predicted = kmeans.labels_

# calculates the frequency of words
from nltk.probability import FreqDist
sent_corp = ''
for o in range(len(corpus)):
  sent_corp = sent_corp + ''.join(corpus[o])
sent_words = nltk.word_tokenize(sent_corp)

from sklearn.feature_extraction.text import CountVectorizer
vp = CountVectorizer()
freq = FreqDist(sent_words)
print(freq)

data['clusters'] = model.labels_

# converting the data into csv 
nat_anth = pd.DataFrame(data)
nat_anth.to_csv('nat_anth.csv')
print("done")