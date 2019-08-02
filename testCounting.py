import numpy as np
import docx
import os
import spacy
from  nltk import everygrams, word_tokenize
from nltk.corpus import stopwords
import fr_core_news_md
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nlp = fr_core_news_md.load()
np.set_printoptions(threshold=np.nan)

def makeDict(a,b):
    return dict(zip(a, b))

def sortDict(a):
    return dict(sorted(a.items(), key=lambda x : x[1], reverse=True))

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text.lower())
    return '\n'.join(fullText)##.replace("\xa0", " ").replace("\n", " ")

def _countVectorizer(text, gram=1):
    vectorizer = CountVectorizer(ngram_range=(gram,gram)) # , stop_words=set(stopwords.words('english'))
    if type(text) == list:
        X = vectorizer.fit_transform(text)
    else:
        X = vectorizer.fit_transform([text])
    return {k: v for k, v in sortDict(dict(zip(vectorizer.get_feature_names(), X.toarray()[0]))).items() if v != 1}

def _tfidfVectorizerZZ(text, gram=1):
    vectorizer = TfidfVectorizer(ngram_range=(gram,gram)) # , stop_words=set(stopwords.words('french'))
    if type(text) == list:
        X = vectorizer.fit_transform(text)
    else:
        X = vectorizer.fit_transform([text])
    return sortDict(dict(zip(vectorizer.get_feature_names(), X.toarray()[0])))

debut = time.time()
path = "X:/Projets/Intelligence Artificielle/Workspace/Data/"
directory = os.listdir(path)
corpus = []
for file in directory:
    try:
        corpus.append(getText(path + file))
    except:
        print("Error file " + file)
print((time.time() - debut) / 60)

debut = time.time()
allCountVectorizer1 = _countVectorizer(corpus, gram=1)
allCountVectorizer2 = _countVectorizer(corpus, gram=2)
allCountVectorizer3 = _countVectorizer(corpus, gram=3)
print((time.time() - debut) / 60)

debut = time.time()
allTFIDFVectorizer1 = _tfidfVectorizerZZ(corpus, gram=1)
allTFIDFVectorizer2 = _tfidfVectorizerZZ(corpus, gram=2)
allTFIDFVectorizer3 = _tfidfVectorizerZZ(corpus, gram=3)
print((time.time() - debut) / 60)


