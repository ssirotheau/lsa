#######################################################################
#######################################################################
# importação de pacotes

import sklearn
import sys,os,time
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
warnings.filterwarnings("ignore", category=DeprecationWarning,module="pandas", lineno=570)
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
decimal.setcontext(decimal.Context(prec=4))
D=decimal.Decimal
import collections as col
#######################################################################
#######################################################################
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
        modelo0.append([XR[i][1],XR[i][2][0]])

#print(modelo0[0][1])
#print(len(modelo0));exit(1)

random.shuffle(modelo0)
modelo0 = modelo0[:3]

#print(modelo0[0]);
#print(len(modelo0));exit(1)

# Modelo1

modelo1 = []
for i in range(len(XR)):
    if XR[i][1] == 1:
        modelo1.append([XR[i][1],XR[i][2][0]])

#print(modelo1[0])
#print(len(modelo1));exit(1)

random.shuffle(modelo1)
modelo1b = modelo1[:10]
modelo1t = modelo1[10:]

#print(modelo1b[0])
#print(modelo1t[0])
#print(len(modelo1b))
#print(len(modelo1t));exit(1)

# Modelo2

modelo2 = []
for i in range(len(XR)):
    if XR[i][1] == 2:
        modelo2.append([XR[i][1],XR[i][2][0]])

#print(len(modelo2))

random.shuffle(modelo2)
modelo2b = modelo2[:10]
modelo2t = modelo2[10:]

#print(modelo2b[0])
#print(modelo2t[0])
#print(len(modelo2b))
#print(len(modelo2t));exit(1)

# Modelo3

modelo3 = []
for i in range(len(XR)):
    if XR[i][1] == 3:
        modelo3.append([XR[i][1],XR[i][2][0]])

#print(len(modelo3))

random.shuffle(modelo3)
modelo3b = modelo3[:10]
modelo3t = modelo3[10:]

#print(modelo3b[0])
#print(modelo3t[0])
#print(len(modelo3b))
#print(len(modelo3t));exit(1)

# Modelo4

modelo4 = []
for i in range(len(XR)):
    if XR[i][1] == 4:
        modelo4.append([XR[i][1],XR[i][2][0]])

#print(len(modelo4))

random.shuffle(modelo4)
modelo4b = modelo4[:10]
modelo4t = modelo4[10:]

#print(modelo4b[0])
#print(modelo4t[0])
#print(len(modelo4b))
#print(len(modelo4t));exit(1)


# Modelo5

modelo5 = []
for i in range(len(XR)):
    if XR[i][1] == 5:
        modelo5.append([XR[i][1],XR[i][2][0]])

#print(len(modelo5))

random.shuffle(modelo5)
modelo5b = modelo5[:10]
modelo5t = modelo5[10:]

#print(modelo5b[0])
#print(modelo5t[0])
#print(len(modelo5b))
#print(len(modelo5t));exit(1)


base_treina = modelo0 + modelo1b + modelo2b + modelo3b + modelo4b + modelo5b
#print(len(base_treina))
#print(base_treina[0])

base_teste = modelo1t + modelo2t + modelo3t + modelo4t + modelo5t
#print(base_teste[0][1])
#print(len(base_teste));exit(1)

notas_treina=[]
for i in range(0,len(base_treina)): notas_treina.append(base_treina[i][0])

#print(len(notas_treina))
#print(notas_treina);exit(1)

texto_treina=[]
for i in range(0,len(base_treina)): texto_treina.append(base_treina[i][1])

#print(len(texto_treina))
#print(texto_treina[0])

notas_teste=[]
for i in range(0,len(base_teste)): notas_teste.append(base_teste[i][0])

#print(len(notas_teste))
#print(notas_teste);exit(1)

texto_teste=[]
for i in range(0,len(base_teste)): texto_teste.append(base_teste[i][1])

#print(len(texto_teste))
#print(texto_teste[0])

base_notas=notas_treina+notas_teste

#print(len(base_notas))
#print(base_notas);exit(1)

base=texto_treina+texto_teste

#print(len(base))

#print(base[0])
#print(base[53]);exit(1)

#######################################################################
#######################################################################
# 1) Construção da Matriz termo-documento

words=['a','agora','ainda','alguém','algum','alguma','algumas','alguns','ampla','amplas','amplo','amplos','ante','antes','ao','aos','após','aquela','aquelas','aquele','aqueles','aquilo','as','até','através','cada','coisa','coisas','com','como','contra','contudo','da','daquele','daqueles','das','de','dela','delas','dele','deles','depois','dessa','dessas','desse','desses','desta','destas','deste','deste','destes','deve','devem','devendo','dever','deverá','deverão','deveria','deveriam','devia','deviam','disse','disso','disto','dito','diz','dizem','do','dos','e','é','ela','elas','ele','eles','em','enquanto','entre','era','essa','essas','esse','esses','esta','está','estamos','estão','estas','estava','estavam','estávamos','este','estes','estou','eu','fazendo','fazer','feita','feitas','feito','feitos','foi','for','foram','fosse','fossem','grande','grandes','há','isso','isto','já','la','lá','lhe','lhes','lo','mas','me','mesma','mesmas','mesmo','mesmos','meu','meus','minha','minhas','muita','muitas','muito','muitos','na','não','nas','nem','nenhum','nessa','nessas','nesta','nestas','ninguém','no','nos','nós','nossa','nossas','nosso','nossos','num','numa','nunca','o','os','ou','outra','outras','outro','outros','para','pela','pelas','pelo','pelos','pequena','pequenas','pequeno','pequenos','per','perante','pode','pude','podendo','poder','poderia','poderiam','podia','podiam','pois','por','porém','porque','posso','pouca','poucas','pouco','poucos','primeiro','primeiros','própria','próprias','próprio','próprios','quais','qual','quando','quanto','quantos','que','quem','são','se','seja','sejam','sem','sempre','sendo','será','serão','seu','seus','si','sido','só','sob','sobre','sua','suas','talvez','também','tampouco','te','tem','tendo','tenha','ter','teu','teus','ti','tido','tinha','tinham','toda','todas','todavia','todo','todos','tu','tua','tuas','tudo','última','últimas','último','últimos','um','uma','umas','uns','vendo','ver','vez','vindo','vir','vos','vós']


def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):
    txt = str(text)
    #txt = re.sub(r'[^A-Za-z\s]',r' ',txt)
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in words])
    if stemming:
        st = RSLPStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])
    #if lemmatization:
    #    wordnet_lemmatizer = WordNetLemmatizer()
    #    txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])
    return txt

base_cst=[]
for i in range(0,len(base)):
    base_cst.append(cleanData(base[i], lowercase = True,remove_stops = True,stemming = True))

#print(base_cst[1]);exit(1)

base=base_cst

#print(base[0])
#print(len(base));exit(1)

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

# print(p_bi,q_bi);exit(1)

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
cosseno_treina = []
for i in range(0, 53):
    L = []
    for j in range(0, 53):
        L.append(round(Cosseno(Ak[:,j], Ak[:,i]), 2))
    cosseno_treina.append(L)

#print(len(cosseno_treina))

#print(len(cosseno_treina[0]))

#print(cosseno_treina[1])

cosseno_teste = []
for i in range(53, qk):
    L = []
    for j in range(0, 53):
        L.append(round(Cosseno(Ak[:,j], Ak[:,i]), 2))
    cosseno_teste.append(L)

#print(len(cosseno_teste))

#print(len(cosseno_teste[0]))

#print(cosseno_teste[0]);exit(1)

cosseno=cosseno_treina+cosseno_teste

#print(len(cosseno))

#print(len(cosseno[1]))

#print(cosseno[1]);exit(1)

I=list(range(53))

#print(I)

ZIP=[]
for i in range(0,len(cosseno)):
    L=[]
    for j in range(0,53):
        L.append([cosseno[i][j],I[j]])
    ZIP.append(L)

#print(len(ZIP))

#print(len(ZIP[1]))

#print(ZIP[1])

for i in range(len(ZIP)): ZIP[i].sort()
for i in range(len(ZIP)): ZIP[i].reverse()

#print(ZIP[1])

kkk=3
ZIPk=[]
for i in range(0,len(ZIP)):ZIPk.append(ZIP[i][0:kkk])

#print(len(ZIPk))
#print(len(ZIPk[1]))
#print(ZIPk[1])

UNZIPk=[]
for i in range(0,len(ZIPk)):
    L=[]
    for j in range(0,kkk):
        L.append(ZIPk[i][j][1])
    UNZIPk.append(L)

#print(len(UNZIPk))
#print(len(UNZIPk[1]))
#print(UNZIPk[1])

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


NOT=[]
for i in range(len(UNZIPk)):
    NOT.append(med(UNZIPk[i]))

#print(len(NOT))
#print(NOT[1])
#print(len(NOT[1]))


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

#print(MED[1])

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

#print(MOD[1])

#print(len(MOD))


############################################################################

#print(len(base_notas))


def erro(L1,L2):return(sum(list(map(lambda x,y:abs((x-y)),L1,L2)))/len(L1))

def acuracia(L1,L2): return ((5-erro(L1,L2))/5)*100


acc1=acuracia(base_notas,MED)
acc2=acuracia(base_notas,MOD)

print(acc1)

print(acc2)
