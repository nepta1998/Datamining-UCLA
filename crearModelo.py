import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.externals import joblib


df = pd.read_csv('Boletines1.csv')
df1 = pd.read_csv('prueba.txt')
df = df[pd.notnull(df['Supervision'])]

#===================================BALANCEANDO LOS DATOS==========================================================================================================
majority = df[df["Supervision"] == 0]
minority = df[df["Supervision"] == 1].sample(n=len(majority), replace=True)
minority1 = df[df["Supervision"] == 3].sample(n=len(majority), replace=True)
dfr = pd.concat([majority, minority, minority1], axis=0)

#dfr.reset_index().to_csv('boletineEx.csv',header=True,index=False)
#===================================================================================================================================================================
#=====================================EXPLORACION DE LOS DATOS=======================================================================================================
plt.figure(figsize=(10, 4))
df.Supervision.value_counts().plot(kind='pie')
plt.show()

plt.figure(figsize=(10, 4))
dfr.Supervision.value_counts().plot(kind='bar')
plt.show()
#======================================================================================================================================================================
my_Supervision = ['Incorrecto', 'Correcto', 'Dudoso']

#========================LIMPIEZA Y PROCESAMINETO DE LOS DATOS ========================================================================================================
REPLACE_BY_SPACE_RE = re.compile('[/()\{\}[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('spanish'))


def clean_text(text):
  
    text = text.lower()  # COLOCAR TODO EL TEXTO EN MINUSCULA
    # REMPLAZAMOS SIMBOLOS POR ESPACIO
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # ELIMINAMOS SIMBOLOS 
    text = BAD_SYMBOLS_RE.sub('', text)
    # ELIMINAMOS LAS STOPWORDS 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def transformarTexto(text):
    corpus = []
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    #ps = PorterStemmer()
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    corpus.append(text)
    return corpus


dfr['Texto'] = dfr['Texto'].apply(clean_text)
df1['Texto'] = df1['Texto'].apply(clean_text)

X = dfr.Texto
y = dfr.Supervision
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#===========================================================================================================================================================================================
#==============================================CREACION DEL CLASIFICADOR MULTINOMIAL DE BAYES==================================================================================================
nb = MultinomialNB()
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
'''nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
               ])'''
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(count_vect.transform(X_test))
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=my_Supervision))

print(nb.predict(count_vect.transform(df1['Texto'])))
#print(nb.predict(count_vect.transform(transformarTexto("de 16 de diciembre de 1954, y concordantes de su reglamento (decreto de 26 de abril de 1957), esta demarcación de carreteras del estado"))))
#=======================================================GUARDAR CLASIFICADOR=========================================================================================================================
'''joblib.dump(nb,'modelo.sav')
joblib.dump(count_vect,'count_vect.sav')'''