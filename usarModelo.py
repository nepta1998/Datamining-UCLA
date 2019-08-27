import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pickle

nb = joblib.load('modelo.sav')
count_vect = joblib.load('count_vect.sav')

REPLACE_BY_SPACE_RE = re.compile('[/()\{\}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('spanish'))

def transformarTexto(text):
    corpus = []
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    #ps = PorterStemmer()
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    corpus.append(text)
    return corpus


def prediccion(texto):
    q = count_vect.transform(transformarTexto(texto))
    predict=nb.predict(q)
    return predict


