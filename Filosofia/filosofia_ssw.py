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

documents_ssw=[]
for i in range(0,len(documents)):
    documents_ssw.append(cleanData(documents[i], lowercase = True,remove_stops = True))

#print(documents_ssw[1]);exit(1)

documents=documents_ssw

#print(documents[1])
#print(len(documents));exit(1)

####Unigramas

vectorizer_uni = CountVectorizer(min_df = 1,ngram_range=(1,1)) #,stop_words = words,analyzer = 'word')


dtm = vectorizer_uni.fit_transform(documents)
df=pd.DataFrame(dtm.toarray(),index=documents,columns=vectorizer_uni.get_feature_names()).head(190)
m=df.values.T.tolist()
matriz=np.matrix(m)
p,q=matriz.shape

#print(p,q);exit(1)

vectorizer_bi = CountVectorizer(min_df = 1,ngram_range=(2,2)) #,stop_words = words,analyzer = 'word')

dtm_bi = vectorizer_bi.fit_transform(documents)
df_bi=pd.DataFrame(dtm_bi.toarray(),index=documents,columns=vectorizer_bi.get_feature_names()).head(190)
m_bi=df_bi.values.T.tolist()
matriz_bi=np.matrix(m_bi)
p_bi,q_bi=matriz_bi.shape

#print(p_bi,q_bi);exit(1)

t001='Dimensões da matriz termo-documento: '

ta='%5.2f' %p, '%5.2f' %q,'%5.2f' %p_bi,'%5.2f' %q_bi

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

t002='Dimensão do espaço semantico - k=14'

k=14

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

pk,qk = Ak.shape

#print(pk,qk);exit(1)

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
    distancia.append(round(dist_euclidiana(Ak[:,0],Ak[:,i]),1))

#print(distancia);exit(1)

distancia_bi=[]
for i in range(0,q):
    distancia_bi.append(round(dist_euclidiana(Ak_bi[:,0],Ak_bi[:,i]),1))

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

def norm(L):
    m=min(L);M=max(L);Mm=M-m
    return list(map(lambda x:0.5*(x-m)/Mm,L))

# discretização: dividir intervalo [0,0.5] em seis partes iguais

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

t01='Acurácia unigrama - cosseno vetorial: '

acc1=round(acuracia(norm(cosseno),norm(hum)),2)

#int(acc1);exit(1)

t1='%5.2f' %acc1

#######################################################################################

t02='Acurácia bigrama - cosseno vetorial: '

acc2=round(acuracia(norm(cosseno_bi),norm(hum)),2)

#print(acc2);exit(1)

t2='%5.2f' %acc2

##########################################################################################

t03='Acurácia unigrama - distancia: '

acc3=round(acuracia(norm(distancia),norm(hum)),2)

#print(acc3);exit(1)

t3='%5.2f' %acc3

############################################################################################

t04='Acurácia bigrama - distancia: '

acc4=round(acuracia(norm(distancia_bi),norm(hum)),2)

#print(acc4);exit(1)

t4='%5.2f' %acc4

############################################################################################

t05='Acurácia unigrama + bigrama - cosseno vetorial: '

x1=np.column_stack((cosseno,cosseno_bi))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

cosseno_uni_bi=[P1[1]*cosseno[i]+P1[2]*cosseno_bi[i]+P1[0] for i in range(len(cosseno))]

acc5=round(acuracia(norm(cosseno_uni_bi),norm(hum)),2)

#print(acc5);exit(1)

t5='%5.2f' %acc5

#############################################################################################

t06='Acurácia unigrama + bigrama - distancia: '

x1=np.column_stack((distancia,distancia_bi))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

distancia_uni_bi=[P1[1]*distancia[i]+P1[2]*distancia_bi[i]+P1[0] for i in range(len(distancia))]

acc6=round(acuracia(norm(distancia_uni_bi),norm(hum)),2)

#print(acc6);exit(1)

t6='%5.2f' %acc6

#############################################################################################

t07='Acurácia unigrama  - cosseno vs distancia: '

x1=np.column_stack((cosseno,distancia))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

cossenovsdistancia=[P1[1]*cosseno[i]+P1[2]*distancia[i]+P1[0] for i in range(len(distancia))]

acc7=round(acuracia(norm(cossenovsdistancia),norm(hum)),2)

#print(acc7);exit(1)

t7='%5.2f' %acc7

#############################################################################################

t08='Acurácia bigrama  - cosseno vs distancia: '

x1=np.column_stack((cosseno_bi,distancia_bi))
x1=sm.add_constant(x1,prepend=True)

P1=sm.OLS(hum,x1).fit().params

cossenovsdistancia_bi=[P1[1]*cosseno_bi[i]+P1[2]*distancia_bi[i]+P1[0] for i in range(len(distancia_bi))]

acc8=round(acuracia(norm(cossenovsdistancia_bi),norm(hum)),2)

#print(acc8);exit(1)

t8='%5.2f' %acc8

#############################################################################################

arq = open('/home/jcarlos/Documentos/jc/AvaliacaoAutomatica/Avaliacao/LSA/Filosofia/filosofia.txt', 'a')

arq.write(str(t001)+'\n')
arq.write(str(ta)+'\n')

arq.write(str(t002)+'\n')

arq.write(str(t01)+'\n')
arq.write(str(t1)+'\n')
arq.write(str(t02)+'\n')
arq.write(str(t2)+'\n')
arq.write(str(t03)+'\n')
arq.write(str(t3)+'\n')
arq.write(str(t04)+'\n')
arq.write(str(t4)+'\n')
arq.write(str(t05)+'\n')
arq.write(str(t5)+'\n')
arq.write(str(t06)+'\n')
arq.write(str(t6)+'\n')
arq.write(str(t07)+'\n')
arq.write(str(t7)+'\n')
arq.write(str(t08)+'\n')
arq.write(str(t8)+'\n')

arq.close()
