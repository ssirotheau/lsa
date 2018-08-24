__author__ = 'jcarlos'

from statistics import mode
import random
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

import nltk
nltk.download()


'''
kkk=4
def nota(n):return n // 2
def med(Z):
    z,nota = zip(* Z[0:kkk])
    nota=map(lambda x:x//2, nota)
    return nota

L = [(0.20000000000000001, 7), (0.20999999999999999, 6), (0.46000000000000002, 0), (0.56999999999999995, 1), (0.60999999999999999, 9), (0.62, 4), (0.69999999999999996, 2), (0.76000000000000001, 5), (0.93000000000000005, 8), (0.96999999999999997, 3), (1.0, 10), (1.0, 11)]

L.reverse()

#print(L)
#print (list(med(L)))

#L1=list(med(L))
#L1.count()
#print (mode(med(L)))

def makealist(row, column):
    list = [[random.randint(0, 100) for x in range(row)] for y in range(column)]
    return list

listA=makealist(3,5)
#print(listA)

def printlist(list):
    for item in list:
        print(item[0], ', '.join(map(str, item[1:])))
    return list

#listB=printlist(listA)
#print(listB)

def labelValues(list):
    numDict = {}
    values = []
    oddEven = " ";
    keys = lambda list: [item for sublist in list for item in sublist]
    for key in keys:
        if key % 2 == 0:
            oddEven = "Even"
            values.append(oddEven)
        else:
            oddEven = "Odd"
            values.append(oddEven)
    print(values)
    numDict = dict(zip(keys, values))
    print(numDict)
    return values


#list = makealist(10, 10)

#print(labelValues(list))

s = 'abc'
t = (10, 20, 30)

#printlist(zip(s,t))

s = 'abc'
t = (10, 20, 30)
u = (-5, -10, -15)

#print(list(zip(s,t,u)))

names = ['Tom', 'Dick', 'Harry']
ages = [50, 35, 60]

#print(dict(zip(names, ages)))

#print([(s[i], t[i]) for i in range(len(s))])

s = 'abcd'
t = (10, 20, 30)

#print(sorted((s,t), key=len))


#print([(s[i], t[i]) for i in range(len(sorted((s,t), key=len)[0]))])

def shortest_sequence_range(*args):
        return range(len(sorted(args, key=len)[0]))

#print([(s[i], t[i]) for i in shortest_sequence_range(s,t) ])


g = ((s[i], t[i]) for i in shortest_sequence_range(s,t) )

#for item in g: print(item)

list_a=[0,1,2,3,4,5,6,7,8,9,10,11]
list_b=['a','b','c','d','e','f','g','h','i','j','k','l']

ZIP=zip(list_b,list_a)

ZZZ=list(ZIP)

#print(ZZZ)
L=list(range(12))
print(L)

def moda(x):
    newrow = np.empty((1, 2))
    newrow[0,] = [x[0], 0]
    i = 0
    maximo = 0
    vmaximo = []
    while i < np.shape(newrow)[0]:
        j = 0
        while j < len(x):
            if newrow[i, 0] == x[j]:
                newrow[i, 1] = newrow[i, 1] + 1
                x.pop(j)
            else:
                if x[j] not in newrow[:, 0]:
                    newrow = np.vstack([newrow, [x[j], 0]])
                j = j + 1
        if maximo <= newrow[i, 1]:
            if maximo == newrow[i, 1]:
                vmaximo = np.append(vmaximo, newrow[i, 0])
                print(vmaximo)
            else:
                vmaximo = newrow[i, 0]
                maximo = newrow[i, 1]
        i = i + 1
    #return (newrow, vmaximo, maximo)
    return vmaximo[0]

x = [1, 5, 3, 4, 6, 7, 3, 9, 10, 5]
out = moda(x)
print(out)

def repete(L):
    repeticoes = 0
    for i in L:
        aparicoes = L.count(i)
    if aparicoes > repeticoes:
        repeticoes = aparicoes
    return repeticoes

#print(repete([5,4,2,4]));exit(1)

def moda(L):
    for i in L:
        aparicoes = L.count(i)
    if aparicoes == repete(L):
        return aparicoes

print(moda([5,5,4,2]));exit(1)

import collections as col
import re

#2 frequencia da distribuição
a = [1,1,5,5,1,1,2,2,2,2,2,2,5,5,2,1,1,1,5,4,5,5]
b = [1,1,0,0,1,1,2,7,7,7,9,9,2,2,2,3,3,4,5,5,5,5,5]

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

#print(moda(a))
#print(moda(b))

L1=[1,2,3,4]
L2=[5,6,7,8]

L=zip(L1,L2)

H=list(zip(*L))

print(list(H[1]))
'''