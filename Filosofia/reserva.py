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

#######################################################################
#######################################################################

# Corpus de pesquisa

XR=[]
with open('filosofia.csv', 'r',encoding='UTF-8') as f:
    XY = csv.reader(f)
    for Y in XY:XR.append([int(Y[0]),float(Y[1]),[str(Y[2])]])

#print(XR[0][2][0])

#print(len(XR));exit(1)

#############################################################################################
#############################################################################################
# Resposta de referência

t0='Resposta de referecia - 4 respostas'

modelo=[]
for i in range(len(XR)):
    if XR[i][1]==5:
        modelo.append(XR[i][2][0])

random.shuffle(modelo)
modelo="".join(modelo[:4])

#print(modelo);exit(1)

#######################################################################
#######################################################################
# 1) Construção da Matriz termo-documento

documents=[]
for i in range(0,len(XR)):documents.append(str(XR[i][2][0]))

documents.insert(0,modelo)

#print(documents[1])
#print(len(documents));exit(1)

####Unigramas

vectorizer_uni = CountVectorizer(min_df = 1,ngram_range=(1,1)) #,stop_words = words,analyzer = 'word')


dtm = vectorizer_uni.fit_transform(documents)
df=pd.DataFrame(dtm.toarray(),index=documents,columns=vectorizer_uni.get_feature_names()).head(190)
m=df.values.T.tolist()
matriz=np.matrix(m)
p,q=matriz.shape

#print(p,q)

vectorizer_bi = CountVectorizer(min_df = 1,ngram_range=(2,2)) #,stop_words = words,analyzer = 'word')

dtm_bi = vectorizer_bi.fit_transform(documents)
df_bi=pd.DataFrame(dtm_bi.toarray(),index=documents,columns=vectorizer_bi.get_feature_names()).head(190)
m_bi=df_bi.values.T.tolist()
matriz_bi=np.matrix(m_bi)
p_bi,q_bi=matriz_bi.shape

#print(p_bi,q_bi)

#######################################################################################################
#######################################################################################################
# 2) pessagem da matriz termo-documento pelo esquema de ponderação tf-idf

def TfIdf(matriz):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(matriz)
    return tfidf.toarray()

matriz_peso=TfIdf(matriz)

matriz_peso_bi=TfIdf(matriz_bi)

#########################################################################################################
#########################################################################################################
# 3) cálculo da SVD

U,Sigma,Vt=linalg.svd(matriz_peso)

u1,u2=U.shape
v1,v2=Vt.shape
s=len(Sigma)

#print(u1,u2)
#print(v1,v2)
#print(s);exit(1)

U_bi,Sigma_bi,Vt_bi=linalg.svd(matriz_peso_bi)

u1_bi,u2_bi=U_bi.shape
v1_bi,v2_bi=Vt_bi.shape
s_bi=len(Sigma_bi)

#print(s_bi);exit(1)

########################################################################
########################################################################

# 4) redução para o espaço semantico

t00='Dimensão do espaço semantico - k=4'

k=2

S=np.array(Sigma[0:k])
S_bi=np.array(Sigma_bi[0:k])


Sk=np.diag(S)
Sk_bi=np.diag(S_bi)

#print(Sk);exit(1)
#print(Sk_bi);exit(1)

Vtk=Vt[:,0:k]
Vtk_bi=Vt_bi[:,0:k]
Vtkt=Vtk.transpose()
Vtkt_bi=Vtk_bi.transpose()

r,s=Vtkt.shape

Ak=np.dot(Sk,Vtkt)
Ak_bi=np.dot(Sk_bi,Vtkt_bi)

p,q = Ak.shape

#print(p,q);exit(1)

##########################################################################
##########################################################################

# 5) medidas de similaridade

def Cosseno(u,v):
    return abs(np.dot(u,v))/(np.linalg.norm(u)*np.linalg.norm(v))

def dist_euclidiana(u,v):
	u, v = np.array(u), np.array(v)
	diff = u - v
	quad_dist = np.dot(diff, diff)
	return math.sqrt(quad_dist)

##########################################################################
##########################################################################
# pontuações - cosseno vetorial

cosseno=[]
for i in range(0,q):
    cosseno.append(Cosseno(Ak[:,0],Ak[:,i]))

#print(cosseno);exit(1)

cosseno_bi=[]
for i in range(0,q):
    cosseno_bi.append(Cosseno(Ak_bi[:,0],Ak_bi[:,i]))

#print(cosseno_bi);exit(1)

# pontuações - cosseno vetorial

distancia=[]
for i in range(0,q):
    distancia.append(round(dist_euclidiana(Ak[:,0],Ak[:,i])/2,1))

#print(distancia);exit(1)

distancia_bi=[]
for i in range(0,q):
    distancia_bi.append(round(dist_euclidiana(Ak_bi[:,0],Ak_bi[:,i])/2,1))

#print(distancia_bi);exit(1)

##########################################################################
##########################################################################
# 7) pontuaçõoes do avaliador humano

hum=[5]
for i in range(0,len(XR)):hum.append(XR[i][1])

#print(hum);exit(1)
#print(len(hum));exit(1)
################################################################################
################################################################################
# 8) Transformaçao dos dados

## norma unitária

def media(L): return sum(L)/len(L)

def d(L):
    somaq=[]
    for i in range(0,len(L)):
        somaq.append((L[i]-media(L))**2)
    return (sum(somaq)**(1/2))

def norm_uni(L):
    N=[]
    for i in range(0,len(L)):
        N.append(abs(L[i]-media(L))/d(L))
    return N

## norma max-min
def norm(L):
    m=min(L);M=max(L);Mm=M-m
    return list(map(lambda x:0.5*(x-m)/Mm,L))

# norma para dados discretos
def normd(L,m,M):
    Mm=M-m
    return list(map(lambda x:(x-m)/Mm,L))

#print(norm(cosseno));exit(1)

# discretização: dividir intervalo [0,1] em seis partes iguais

def discretizar(L):
    D=[]
    for i in range(0,len(L)):
        if L[i]<=1/12:
            D.append(0.0)
        elif L[i]<=2/12:
            D.append(1.0)
        elif L[i]<=3/12:
            D.append(2.0)
        elif L[i]<=4/12:
            D.append(3.0)
        elif L[i]<=5/12:
            D.append(4)
        else:
            D.append(5.0)
    return D

#################################################################################
################################################################################
# 8) erro médio e acurácias

def erro(L1,L2):return(sum(list(map(lambda x,y:abs((x-y)),L1,L2)))/len(L1))

def acuracia(L1,L2): return 1-erro(L1,L2)

def acuracia2(L1,L2): return ((5-erro(L1,L2))/5)*100

##################################################################################
##################################################################################

# 9) regressão linear

def quadrado(L):return [x**2 for x in L]

def produto(L1,L2):
    prod=[]
    for i in range(0,len(L1)):
        prod.append(L1[i]*L2[i])
    return prod

def coef_angular(L1,L2):return (len(L1)*sum(produto(L1,L2))-sum(L1)*sum(L2))/(len(L1)*sum(quadrado(L1))-(sum(L1)**2))

def intercepto(L1,L2): return (sum(L2)-coef_angular(L1,L2)*sum(L1))/len(L1)


##############################################################################################
##############################################################################################

t01='Acurácia unigrama - cosseno vetorial: '

a=coef_angular(cosseno,norm(hum))
b=intercepto(cosseno,norm(hum))

cossenor=[round(a,2)*cosseno[i]+round(b,2) for i in range(len(cosseno))]

acc1=round(acuracia(norm(cosseno),norm(hum)),2)

print(acc1);exit(1)

t1='%5.2f' %acc1

#######################################################################################

t02='Acurácia bigrama - cosseno vetorial: '

a=coef_angular(cosseno_bi,hum)
b=intercepto(cosseno_bi,hum)

cosseno_bir=[a*cosseno_bi[i]+b for i in range(len(cosseno_bi))]

#print(cosseno_bir);exit(1)

acc2=round(acuracia2(cosseno_bir,hum),2)

#print(acc2);exit(1)

t2='%5.2f' %acc2

##########################################################################################

t03='Acurácia unigrama - distancia: '

a=coef_angular(distancia,hum)
b=intercepto(distancia,hum)

distanciar=[a*distancia[i]+b for i in range(len(distancia))]

acc3=round(acuracia2(distanciar,hum),2)

#print(acc3);exit(1)

t3='%5.2f' %acc3

############################################################################################

t04='Acurácia bigrama - distancia: '

a=coef_angular(distancia_bi,hum)
b=intercepto(distancia_bi,hum)

distancia_bir=[a*distancia_bi[i]+b for i in range(len(distancia))]

acc4=round(acuracia2(distancia_bir,hum),2)

#print(acc4);exit(1)

t4='%5.2f' %acc4

############################################################################################

t05='Acurácia unigrama + bigrama - cosseno vetorial: '

x1=np.column_stack((cosseno,cosseno_bi))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

cosseno_uni_bi=[P1[1]*cosseno[i]+P1[2]*cosseno_bi[i]+P1[0] for i in range(len(cosseno))]

acc5=round(acuracia2(cosseno_uni_bi,hum),2)

#print(acc5);exit(1)

t5='%5.2f' %acc5

#############################################################################################

t06='Acurácia unigrama + bigrama - distancia: '

x1=np.column_stack((distancia,distancia_bi))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

distancia_uni_bi=[P1[1]*distancia[i]+P1[2]*distancia_bi[i]+P1[0] for i in range(len(distancia))]

acc6=round(acuracia2(distancia_uni_bi,hum),2)

#print(acc6);exit(1)

t6='%5.2f' %acc6

#############################################################################################

t07='Acurácia unigrama  - cosseno vs distancia: '

x1=np.column_stack((cosseno,distancia))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

cossenovsdistancia=[P1[1]*cosseno[i]+P1[2]*distancia[i]+P1[0] for i in range(len(distancia))]

acc7=round(acuracia2(cossenovsdistancia,hum),2)

#print(acc7);exit(1)

t7='%5.2f' %acc7

#############################################################################################

t08='Acurácia bigrama  - cosseno vs distancia: '

x1=np.column_stack((cosseno_bi,distancia_bi))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

cossenovsdistancia_bi=[P1[1]*cosseno_bi[i]+P1[2]*distancia_bi[i]+P1[0] for i in range(len(distancia_bi))]

acc8=round(acuracia2(cossenovsdistancia_bi,hum),2)

#print(acc8);exit(1)

t8='%5.2f' %acc8

#############################################################################################



