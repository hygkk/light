import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from nltk import bigrams, FreqDist,pos_tag
from nltk.tokenize import word_tokenize
import nltk
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LabelSet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import argparse
import time 
import pdb
import random
'''
SGD（下溢/轮次）
'''

d=100      # 词向量维度
k=2        # 正负样本比例
alpha=0.75 # 负采样随机抽取中的指数
window=4   
learning_rate=0.0001

def n_gram(tokens:list,window:int):
    ngram=[]
    n=len(tokens)
    for i in range(n):
        word1=tokens[i]
        for j in range(-window//2,window//2+1):
            if j==0:
                continue
            if i+j>0 and i+j<n:
                word2=tokens[i+j]
                ngram.append((word1,word2))
    return ngram


def get_data():
    with open ('/data/huyangge/nlptutorial/task04-co-matrix/final_result.json') as f:
        data=json.load(f)
    text=''
    for v in data.values():
        for value in v.values():
            for i in value:
                text+=i['text']+' '
    tokens=[]
    for token in word_tokenize(text):
        if '-' in token:
            token=token.split('-')
            for i in token:
                if i:
                    tokens.append(i)
        elif '—' in token:
            token=token.split('—')
            for i in token:
                if i:
                    tokens.append(i)
        else:
            if i:
                tokens.append(token)
    word_freq=FreqDist(tokens)
    n=len(word_freq)
    n_grams=n_gram(tokens,window)
    matrix=pd.DataFrame(np.zeros((2*n,d)))
    positive={}
    num=0
    for i in n_grams:
        word=i[0]
        context=i[1]
        if word not in positive:
            positive[word]=set()
        if context not in positive[word]:
            positive[word].add(context)
            num+=1
    tokens_set=set(tokens)
    p=[]
    tokens=[]
    base=0
    for value in word_freq.values():
        base+=value**alpha
    for key in word_freq.keys():
        tokens.append(key)
        p.append(word_freq[key]**alpha/base)
    
    df=[]
    for w in tokens_set:
        for c in positive[w]: 
            df.append([w,c,1])
            neg_n=0
            while neg_n<k:
                c=random.choices(tokens)[0]
                if c in positive[w] or c==w:
                    continue
                else:
                    df.append([w,c,0])
                    neg_n+=1
    df=pd.DataFrame(df)
    df.columns=['w','c','+']
    df.to_csv('data.txt')
    

def sigmoid(x):
    return 1/(1+np.exp(-x))


def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def word2vec():
    df=pd.read_csv('data.txt',index_col=0)
    W={}
    for i in df['w'].unique():
        v=np.random.rand(d)
        W[i]=normalize(v)
    C=W
    
    for i in range(len(df)//3):
        w=W[df.loc[3*i,'w']]
        p=C[df.loc[3*i,'c']]
        n=[]
        # pdb.set_trace()
        for j in range(k):
            n.append(C[df.loc[3*i+1+j,'c']])
        
        W[df.loc[3*i,'w']]=normalize(w-learning_rate*(sigmoid(np.dot(p,w))-1)*p+sum([sigmoid(np.dot(item,w))*item for item in n])) # 更新w

        C[df.loc[3*i,'c']]=normalize(p-learning_rate*(sigmoid(np.dot(p,w))-1)*w)   # 更新p
        for j in range(k):  # 更新n
            C[df.loc[3*i+1+j,'c']]=normalize(n[j]-learning_rate*sigmoid(np.dot(n[j],w))*w)
    print(W)










if __name__=='__main__':
    #get_data()
    word2vec()







