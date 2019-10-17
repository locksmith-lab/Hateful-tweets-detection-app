import pandas as pd
import numpy as np

#data
data = pd.read_csv("tweet_data.csv")

#leaning data with regex
import re

data["tweet"] = data["tweet"].apply(lambda t: re.sub(r'(\S*@\S*)|(#\S*)|(\S*\d+\S*)|(https?\S*)|([^a-z ])|(\s?\srt\s\s?)', '', t.lower()))
data["tweet"] = data["tweet"].apply(lambda t: re.sub(r'(\s\w\s)|(\s?\srt\s\s?)', ' ', t))

#split into words
data["tweet"] = data["tweet"].apply(lambda t: t.split())


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
list_rt = ["rt"]

data["tweet"] = data["tweet"].apply(lambda tweet: [word for word in tweet if word not in stop_words and word not in list_rt])

data["tweet"] = data["tweet"].apply(lambda x: " ".join(x))

#define target and data for the model
y = data["class"]
X = data["tweet"]


#pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

NBM_pipe = Pipeline([
        ('tfidf_nbm', TfidfVectorizer()),
            ('NB_multi', MultinomialNB(alpha=0.1))])

model_nbm = NBM_pipe.fit(X,y)


import pickle
from sklearn.externals import joblib

joblib.dump(model_nbm,"tweets.pkl")
