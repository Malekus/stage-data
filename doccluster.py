import numpy as np
import docx
import spacy
import os
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
    return [' '.join(word).lower().strip() for word in list(everygrams(word_tokenize(text, language='french'), min_len=mini, max_len=maxi))]
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
        return {k: v for k, v in sortDict(dict(zip(vectorizer.get_feature_names(), X.toarray()[0]))).items() if v != 1}

def _tfidfVectorizer(text, gram=1):
    vectorizer = TfidfVectorizer(ngram_range=(1,gram)) # , stop_words=set(stopwords.words('french'))
    if type(text) == list:
        X = vectorizer.fit_transform(text)
    else:
        X = vectorizer.fit_transform([text])
    return sortDict(dict(zip(vectorizer.get_feature_names(), X.toarray()[0])))

def preProcessingWords(words):
    q = {}
    posForbidden = ['ADP', 'DET', 'CCONJ', 'X', 'ADV', 'AUX', 'PUNCT', 'NUM', 'VERB', 'PRON', 'SCONJ']
    for mot, count in words.items():
        doc = nlp(mot)
        for token in doc:
            if len(token.text) > 4:
                if len(doc) == 1 and mot not in q.keys() and token.pos_ not in posForbidden + ['ADJ']:
                    q[mot] = count
                if mot not in q.keys() and doc[0].pos_ not in posForbidden + ['ADJ'] and doc[-1].pos_ not in posForbidden:
                    q[mot] = count
    return q


def getThreshold(dict_gram, gram, pourcent):
    r = {}
    for nbGram in range(gram):
        t = [x for x in np.array(list(dict_gram.keys())) if len(x.split()) == nbGram + 1]
        r[nbGram + 1] = int(max([ v for k,v in dict_gram.items() if k in t]) * (100 - pourcent) / 100)
    return r


def bestNgram(ngrams, threshold):
    r = {}
    for key, value in ngrams.items(): 
        if value > threshold[len(key.split())]:
            r[key] = value
    return r

def getAllDocs(pathname):
    corpus = []
    for file in os.listdir(path):
        try:
            corpus.append(DocCluster(pathname + file))
            print(file + " done")
        except:
            print("Error file " + file)
    return corpus


def getAllKeys(self):
    d = {}
    for corpus in monCorpus.texts:
        for key, value in corpus.bestNGrams.items():
            if key in d.keys():
                d[key] += value
            else:
                d[key] = value
    return sortDict(d)


class DocCluster:
    def __init__(self, pathname, gram=3, threshold=80):
        self.pathname = pathname
        self.gram = gram
        self.text = getText(self.pathname)
        self.words = getWords(self.text, 1, self.gram)
        self.ngram = _countVectorizer(self.text, self.gram)
        self.threshold = getThreshold(self.ngram, self.gram, threshold)
        self.bestNGrams = bestNgram(preProcessingWords(self.ngram), self.threshold)


class Corpus:
    def __init__(self, path):
        self.pathname = path
        self.texts = getAllDocs(self.pathname)
        self.allKeys = None
        self.getAllKeys()
        
    def getAllKeys(self):
        d = {}
        for corpus in self.texts:
            for key, value in corpus.bestNGrams.items():
                if key in d.keys():
                    d[key] += 1
                else:
                    d[key] = 1
        self.allKeys = sortDict(d)

path = "X:/Projets/Intelligence Artificielle/Workspace/Data/"
debut = time.time()
monCorpus = Corpus(path)
print((time.time() - debut) / 60)
x = DocCluster("X:/Projets/Intelligence Artificielle/Workspace/Data/AC6_CIR 14_Synthèse01_Workbench 4 Linux_vFI.docx")

def getTF(allKeys, wordCount):
    r = [0] * len(allKeys)
    for word, count in wordCount.items():
        r[list(allKeys.keys()).index(word)] = count / float(len(wordCount))
    return r

# map(lambda p: myFunc(p, additionalArgument), pages)

allKeys = {}
for doc in corpus:
    for newKey in doc.bestNGrams.keys():
        if newKey in allKeys.keys():
            allKeys[newKey] += 1
        else:
            allKeys[newKey] = 1
print(allKeys)

debut = time.time()
allKeys = {}
for doc in monCorpus.texts:
    for newKey in doc.bestNGrams.keys():
        if newKey in allKeys.keys():
            allKeys[newKey] += 1
        else:
            allKeys[newKey] = 1
print(allKeys)
print((time.time() - debut) / 60)

debut = time.time()
allTF = []
for doc in monCorpus.texts:
    allTF.append(getTF(allKeys, doc.bestNGrams))
print((time.time() - debut) / 60)

getTF(allKeys, doc.bestNGrams)
monCorpus.texts[0].bestNGrams


debut = time.time()
allTexte = []
for doc in monCorpus.texts:
    allTexte.append(doc.text)
print((time.time() - debut) / 60)

countVectorizer = CountVectorizer(ngram_range=(2,2))
X = countVectorizer.fit_transform(allTexte)
countVectorizerData = X.toarray()
countVectorizerNames = countVectorizer.get_feature_names()

tfidfVectorizer = TfidfVectorizer(ngram_range=(2,2))
X = tfidfVectorizer.fit_transform(allTexte)
tfidfVectorizerData = X.toarray()
tfidfVectorizerNames = tfidfVectorizer.get_feature_names()
