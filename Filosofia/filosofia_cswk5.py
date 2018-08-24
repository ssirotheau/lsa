#######################################################################
#######################################################################
# importação de pacotes

import sklearn
import sys, os, time
import __future__
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import warnings
# Suppress warnings from pandas library
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas", lineno=570)
from numpy import *
import numpy as np
import sqlite3
from scipy import linalg
from scipy.stats import linregress
from math import cos
import matplotlib.pyplot as plt
import statsmodels.api as sm
import operator
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from unicodedata import normalize

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv
from nltk.corpus import stopwords
from nltk import stem
from nltk.stem import RSLPStemmer
import re
import decimal
from statistics import mode
import collections as col

decimal.setcontext(decimal.Context(prec=4))
D = decimal.Decimal

#######################################################################
#######################################################################

# classe para filtrgagem dos textos

class ProcLing():
    def __init__(self, lista):
        self.lista = lista

    def retiraAcentuacao(self):
        lista_sem_acentos = []
        for l in self.lista:
            try:
                l = normalize('NFKD', l).encode('ASCII','ignore').decode('ASCII')
            except:
                l = l
            lista_sem_acentos.append(l)

        return lista_sem_acentos

    def transformaMinusculas(self):
        lista_minuscula=[]
        for i in range(0,len(self.lista)):
            lista_minuscula.append(self.lista[i].lower())
        return lista_minuscula

    def retiraPontuacao(self):
        lista_sem_pontuacao=[]
        for i in range(0,len(self.lista)):
            lista_sem_pontuacao.append(re.sub(u'[...:,;()!?%&]',' ',self.lista[i]))
        return lista_sem_pontuacao

    def filtrar(self):
        self.lista = self.retiraAcentuacao()
        self.lista = self.transformaMinusculas()
        self.lista = self.retiraPontuacao()

        return self.lista

##########################################################################
##########################################################################
# Corpus de pesquisa

XR=[]
with open('filosofia.csv', 'r',encoding='UTF-8') as f:
    XY = csv.reader(f)
    for Y in XY:XR.append([int(Y[0]),float(Y[1]),ProcLing([str(Y[2])]).filtrar()])

#print(XR[0][2][0])

#print(len(XR));exit(1)

#############################################################################################
#############################################################################################
## Base estratificada

#random.seed(13)
# Modelo0

modelo0 = []
for i in range(len(XR)):
    if XR[i][1] == 0:
        modelo0.append(XR[i][2][0])


random.shuffle(modelo0)
modelo0 = modelo0[:3]

#print(modelo0);
#print(len(modelo0));exit(1)

# Modelo1

modelo1 = []
for i in range(len(XR)):
    if XR[i][1] == 1:
        modelo1.append(XR[i][2][0])

random.shuffle(modelo1)
modelo1 = modelo1[:10]

#print(modelo1[0])
#print(len(modelo1));exit(1)

# Modelo2

modelo2 = []
for i in range(len(XR)):
    if XR[i][1] == 2:
        modelo2.append(XR[i][2][0])

random.shuffle(modelo2)
modelo2 = modelo2[:10]

#print(modelo2[0]);
#print(len(modelo2));exit(1)

# Modelo3

modelo3 = []
for i in range(len(XR)):
    if XR[i][1] == 3:
        modelo3.append(XR[i][2][0])

random.shuffle(modelo3)
modelo3 = modelo3[:10]

#print(modelo3[0])
#print(len(modelo3));exit(1)

# Modelo4

modelo4 = []
for i in range(len(XR)):
    if XR[i][1] == 4:
        modelo4.append(XR[i][2][0])

random.shuffle(modelo4)
modelo4 = modelo4[:10]

#print(modelo4[0])
#print(len(modelo4));exit(1)

# Modelo5

modelo5 = []
for i in range(len(XR)):
    if XR[i][1] == 5:
        modelo5.append(XR[i][2][0])

random.shuffle(modelo5)
modelo5 = modelo5[:10]

#print(modelo5[0])
#print(len(modelo5));exit(1)

base = modelo0 + modelo1 + modelo2 + modelo3 + modelo4 + modelo5
#print(len(base));exit(1)

base_notas = []
for i in range(len(base), len(XR)):
    base.append(str(XR[i][2][0]))
    base_notas.append(XR[i][1])

#print(len(base))
#print(len(base_notas));exit(1)

#######################################################################
#######################################################################
## Construção da matriz termo documento

vectorizer_uni = CountVectorizer(min_df=1, ngram_range=(1, 1))  # ,stop_words = words,analyzer = 'word')

dtm = vectorizer_uni.fit_transform(base)
df = pd.DataFrame(dtm.toarray(), index=base, columns=vectorizer_uni.get_feature_names()).head(190)
m = df.values.T.tolist()

matriz = np.matrix(m)
p, q = matriz.shape

#print(p,q);exit(1)

vectorizer_bi = CountVectorizer(min_df=1, ngram_range=(2, 2))  # ,stop_words = words,analyzer = 'word')

dtm_bi = vectorizer_bi.fit_transform(base)
df_bi = pd.DataFrame(dtm_bi.toarray(), index=base, columns=vectorizer_bi.get_feature_names()).head(190)
m_bi = df_bi.values.T.tolist()
matriz_bi = np.matrix(m_bi)
p_bi, q_bi = matriz_bi.shape

#print(p_bi,q_bi);exit(1)

#######################################################################################################
#######################################################################################################

# pessagem da matriz termo-documento pelo esquema de ponderação tf-idf

def TfIdf(matriz):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(matriz)
    return tfidf.toarray()


matriz_peso = TfIdf(matriz)

matriz_peso_bi = TfIdf(matriz_bi)

#########################################################################################################
#########################################################################################################
# 3) cálculo da SVD

U, Sigma, Vt = linalg.svd(matriz_peso)

u1, u2 = U.shape
v1, v2 = Vt.shape
s = len(Sigma)

# print(u1,u2)
# print(v1,v2)
# print(s);exit(1)

U_bi, Sigma_bi, Vt_bi = linalg.svd(matriz_peso_bi)

u1_bi, u2_bi = U_bi.shape
v1_bi, v2_bi = Vt_bi.shape
s_bi = len(Sigma_bi)

# print(s_bi);exit(1)

########################################################################
########################################################################

# 4) redução para o espaço semantico

k = 4

S = np.array(Sigma[0:k])
S_bi = np.array(Sigma_bi[0:k])

Sk = np.diag(S)
Sk_bi = np.diag(S_bi)

# print(Sk);exit(1)
# print(Sk_bi);exit(1)

Vtk = Vt[:, 0:k]
Vtk_bi = Vt_bi[:, 0:k]
Vtkt = Vtk.transpose()
Vtkt_bi = Vtk_bi.transpose()

r, s = Vtkt.shape

Ak = np.dot(Sk, Vtkt)
Ak_bi = np.dot(Sk_bi, Vtkt_bi)

pk, qk = Ak.shape

#print(pk,qk);exit(1)

##########################################################################
##########################################################################

# 5) medidas de similaridade

def Cosseno(u, v):
    return abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))


def dist_euclidiana(u, v):
    u, v = np.array(u), np.array(v)
    diff = u - v
    quad_dist = np.dot(diff, diff)
    return math.sqrt(quad_dist)

##########################################################################
##########################################################################

cosseno = []
for i in range(53, qk):
    L = []
    for j in range(0, 53):
        L.append(round(Cosseno(Ak[:, j], Ak[:, i]), 2))
    cosseno.append(L)

#print(len(cosseno))

#print(cosseno[1])

#print(cosseno[1])

#print(cosseno[2])

#print(len(cosseno[2]))

I=list(range(135))
ZIP=[]
for i in range(len(cosseno)):
    L=zip(cosseno[i],I)
    ZIP.append(list(L))

#print(ZIP[0])
#print(len(ZIP[0]))
#print(len(ZIP))

for i in range(len(ZIP)): ZIP[i].sort()
for i in range(len(ZIP)): ZIP[i].reverse()

#print(ZIP[0])
#print(len(ZIP))


UNZ=[]
for i in range(0,len(ZIP)):UNZ.append(list(zip(*ZIP[i])))

#print(len(UNZ))
#print(UNZ[0])
#print(len(UNZ[0]))

I=[]
for i in range(0,len(UNZ)):I.append(list(UNZ[i][1]))

#print(len(I))
#print(I[0])
#print(len(I[0]))

kkk = 3
UNZIP=[]
for i in range(0,len(I)):UNZIP.append(I[i][0:kkk])

#print(len(UNZIP))
#print(UNZIP[0])
#print(len(UNZIP[0]))

def med(list):
    L=[]
    for item in list:
        if item <=2:
            L.append(0)
        elif item <=12:
            L.append(1)
        elif item <=22:
            L.append(2)
        elif item <=32:
            L.append(3)
        elif item <=42:
            L.append(4)
        else:
            L.append(5)
    return L

#print(med([0,1,13,12,23,22,33,32,43,42]));exit(1)

NOT=[]
for i in range(len(UNZIP)):
    NOT.append(med(UNZIP[i]))

#print(len(NOT))
#print(NOT[0])
#print(len(NOT[0]))

#############################################################################
def media(L): return sum(L)/len(L)
##############################################################################
def freqDist(a,b):
    ca=col.Counter(a)
    cb=col.Counter(b)
    dom=set(list(ca.keys())+list(cb.keys()))
    vab=[(ca[i],cb[i]) for i in dom]
    return vab
def moda(a):
    ca=col.Counter(a)
    cal=list(zip(list(ca.values()),list(ca.keys())))
    cal.sort(reverse=True)
    print(cal)
    return cal[0][1]

###############################################################################

MED=[]
for i in range(len(NOT)): MED.append(round(media(NOT[i]),2))

#print(MED[0])

#print(len(MED))

def freqDist(a,b):
    ca=col.Counter(a)
    cb=col.Counter(b)
    dom=set(list(ca.keys())+list(cb.keys()))
    vab=[(ca[i],cb[i]) for i in dom]
    return vab
def moda(a):
    ca=col.Counter(a)
    cal=list(zip(list(ca.values()),list(ca.keys())))
    cal.sort(reverse=True)
    return cal[0][1]


MOD=[]
for i in range(len(NOT)): MOD.append(round(moda(NOT[i]),0))

#print(MOD[0])

#print(len(MOD))


############################################################################

#print(len(base_notas))


def erro(L1,L2):return(sum(list(map(lambda x,y:abs((x-y)),L1,L2)))/len(L1))

def acuracia(L1,L2): return ((5-erro(L1,L2))/5)*100


print(MED)
print(len(MED))

print(MOD)
print(len(MOD))

print(base_notas)
print(len(base_notas))

acc1=acuracia(base_notas,MED)
acc2=acuracia(base_notas,MOD)

print(acc1)
print(acc2)
