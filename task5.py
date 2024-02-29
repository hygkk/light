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

'''
SGD（下溢/轮次）
'''

d=300      # 词向量维度
k=2        # 正负样本比例
alpha=0.75 # 负采样随机抽取中的指数
window=4   


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
    tokens=word_tokenize(text) 
    word_freq=FreqDist(tokens)
    n=len(word_freq)
    n_grams=n_gram(tokens,window)
    matrix=pd.DataFrame(np.zeros((2*n,d)))
    positive={}
    negative={}
    for i in n_grams:
        word=i[0]
        context=i[1]
        if word not in positive:
            positive[word]=set()
        if context not in positive[word]:
            positive[word].add(context)
    tokens_set=set(tokens)
    p=np.array([])
    tokens=[]
    base=0
    for value in word_freq.values():
        base+=value**alpha
    for key in word_freq.keys():
        tokens.append(key)
        p=np.append(p,word_freq[key]**alpha/base)


    df=pd.DataFrame(columns=['w','c','+'])
    for w in tokens_set:
        for c in positive[w]: 
            df.loc[len(df)]=[w,c,1]
        neg_n=0
        while neg_n<k:
            c=np.random.choice(tokens,p=p.ravel())
            if c in positive[w] or c==w:
                continue
            else:
                print([w,c])
                df.loc[len(df)]=[w,c,0]
                neg_n+=1
    df.to_csv('data.txt')
    

def word2vec():
    df=pd.read_csv('data.txt')
    print(df)


if __name__=='__main__':
    #get_data()
    word2vec()







