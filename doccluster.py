import numpy as np
import docx
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
        fullText.append(para.text)
    return '\n'.join(fullText)##.replace("\xa0", " ").replace("\n", " ")

def getWords(text, mini=1, maxi=3):
    return [' '.join(word).lower().strip() for word in list(everygrams(word_tokenize(texte, language='french'), min_len=mini, max_len=maxi))]
# [x.text for x in doc if x.text not in ['(', ':',')', ';', '.' ,',','«', '»', '[',']', ' ', '-', '/', '?', '<', '>']]
def countWord(tab):
    allWords = list(set(tab))
    r = {}
    for word in allWords:
        r[word] = tab.count(word)
    return sortDict(r)

def _countVectorizer(text, gram=1):
    vectorizer = CountVectorizer(ngram_range=(1,gram)) # , stop_words=set(stopwords.words('english'))
    if type(text) == list:
        X = vectorizer.fit_transform(text)
    else:
        X = vectorizer.fit_transform([text])
    return sortDict(dict(zip(vectorizer.get_feature_names(), X.toarray()[0])))

def _tfidfVectorizer(text, gram=1):
    vectorizer = TfidfVectorizer(ngram_range=(1,gram)) # , stop_words=set(stopwords.words('french'))
    if type(text) == list:
        X = vectorizer.fit_transform(text)
    else:
        X = vectorizer.fit_transform([text])
    return sortDict(dict(zip(vectorizer.get_feature_names(), X.toarray()[0])))

"""
def _tfidfVectorizer(text, vocabulary):
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    if type(text) == list:
        X = vectorizer.fit_transform(text)
    else:
        X = vectorizer.fit_transform([text])
    return sortDict(dict(zip(vectorizer.get_feature_names(), X.toarray()[0])))
"""

texte = getText("X:/Projets/Intelligence Artificielle/Workspace/Data/AC6_CIR 14_Synthèse01_Workbench 4 Linux_vFI.docx")
mots = getWords(texte, 1, 3)

a = _countVectorizer(texte, 3)


r = []
posForbidden = ['ADP', 'DET', 'CCONJ', 'X', 'ADV', 'AUX', 'PUNCT', 'NUM', 'VERB', 'PRON']
for index, mot in enumerate(a.keys()):
    
    doc = nlp(mot)
    m = []
    for token in doc:
        if len(token.text) > 4:
            if len(doc) == 1 and mot not in r and token.pos_ not in posForbidden:
                """
                print("je suis le mot " + mot)
                print(token.text, token.pos_)
                """
                r.append(mot)
            if mot not in r and doc[0].pos_ not in posForbidden and doc[-1].pos_ not in posForbidden:
                r.append(mot)            



for a, b in a.items():
    print(a, b)

q = {}
posForbidden = ['ADP', 'DET', 'CCONJ', 'X', 'ADV', 'AUX', 'PUNCT', 'NUM', 'VERB', 'PRON']
for mot, count in a.items():
    doc = nlp(mot)
    m = []
    for token in doc:
        if len(token.text) > 4:
            if len(doc) == 1 and mot not in q.keys() and token.pos_ not in posForbidden:
                """
                print("je suis le mot " + mot)
                print(token.text, token.pos_)
                """
                q[mot] = count
            if mot not in q.keys() and doc[0].pos_ not in posForbidden and doc[-1].pos_ not in posForbidden:
                q[mot] = count
return q

r == list(q.keys())

q = preProcessingWords(a)
qq = preProcessingWords(toto)

def preProcessingWords(words):
    q = {}
    posForbidden = ['ADP', 'DET', 'CCONJ', 'X', 'ADV', 'AUX', 'PUNCT', 'NUM', 'VERB', 'PRON']
    for mot, count in words.items():
        doc = nlp(mot)
        for token in doc:
            if len(token.text) > 4:
                if len(doc) == 1 and mot not in q.keys() and token.pos_ not in posForbidden:
                    print(token.text, token.pos_)
                    q[mot] = count
                if mot not in q.keys() and doc[0].pos_ not in posForbidden and doc[-1].pos_ not in posForbidden:
                    q[mot] = count
    return q



toto = _tfidfVectorizer(texte, 3)


class DocCluster:
    def __init__(self, pathname, gram=3):
        self.pathname = pathname
        self.gram = gram
        self.text = getText(self.pathname)
        self.words = getWords(self.text, 1, self.gram)
        self.ngram = _countVectorizer(self.text, self.gram)
        
        

x = DocCluster("X:/Projets/Intelligence Artificielle/Workspace/Data/AC6_CIR 14_Synthèse01_Workbench 4 Linux_vFI.docx")

work = x.ngram

for i in range(1, 3):
    print(i)
len(np.column_stack((list(work.keys()), list(work.values())))[:,0])

np.array(work.keys()) == np.array([x for x in np.array(list(work.keys())) if len(x.split()) == 1])
