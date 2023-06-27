import pandas as pd
import numpy as np
import os, sys, gzip
import argparse, random, pickle
from tqdm import tqdm
from itertools import count
from collections import defaultdict
import csv
import collections
from math import sqrt
from sentence_transformers import SentenceTransformer as st
bert_model_path = "./data/allMiniLML6v2"
dataPath = './data/'

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
        
def getDF(path):
    df = {}
    for i,d in enumerate(parse(path)):
        df[i] = d
    return pd.DataFrame.from_dict(df, orient='index')

def load_data(path):
    s_5core = getDF(path)
    s_users = set(s_5core['reviewerID'].tolist())
    overlapping_users = s_users 
    df = s_5core[s_5core['reviewerID'].isin(overlapping_users)][['reviewerID','asin','overall','reviewText','unixReviewTime']]
    return df

def convert_idx(df):
    uiterator_s = count(0)
    udict_s = defaultdict(lambda: next(uiterator_s))
    [udict_s[user] for user in df["reviewerID"]]
    iiterator_s = count(0)
    idict_s = defaultdict(lambda: next(iiterator_s))
    [idict_s[item] for item in df["asin"]]
    df['uid'] = df['reviewerID'].map(lambda x: udict_s[x])
    df['iid'] = df['asin'].map(lambda x: idict_s[x])
    user_set_s = set(df['uid'])
    item_set_s = set(df['iid'])
    assert len(item_set_s) == len(idict_s)
    user_num_s, item_num_s = len(user_set_s), len(item_set_s)
    print('users %d, items %d, ratings %d.' % (user_num_s, item_num_s, len(df)))
    return user_num_s, item_num_s, dict(udict_s), dict(idict_s), df

def get_documents(df):
    reviews = df['reviewText']
    reviews = [np.array(review)[np.array(review) != -1].tolist() for review in reviews]   
    df = df.copy()
    df['review_idx'] = reviews
    docu_udict = defaultdict(list)
    docu_idict = defaultdict(list)
    for user, item, review in zip(df['uid'], df['iid'], df['review_idx']):
        docu_udict[user].extend(review)
        docu_idict[item].extend(review)
    return docu_udict, docu_idict


def generate_vectors(bert_model_path,data):
    model = st(bert_model_path)
    docu_list = []
    for i in range(len(data)):     
        raw = data[i]
        embeddings = model.encode(raw)
        embeddings = sum(embeddings)
        docu_list.append(embeddings)
    return docu_list

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim
def pearson(v1,v2):
    sum1=sum(v1)
    sum2=sum(v2)
    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])
    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])
    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 0
    return 1.0-num/den

def Matrix_sim(docu_list,Matrix):
    num=1
    num_total=0
    for item1 in range(len(docu_list)):
        for item2 in range(num,len(docu_list)):
            Review = cos_sim(docu_list[int(item1)], docu_list[int(item2)])
            if Review > 0.83:                           
                Matrix[int(item1),int(item2)] = 1
                Matrix[int(item2),int(item1)] = 1
                num_total += 1
        num += 1        

    print('Matrix',Matrix)
    print('num_total',num_total)
    print("sparsity",num_total/(len(docu_list)*len(docu_list)) * 100)
    return Matrix

def save_data(path,df):
    del df['reviewText']
    df.to_csv(path+'/Digital_Music.csv', index=False)
    return df

df=load_data(dataPath+'reviews_Digital_Music_5.json.gz')
user_num, item_num,udict,idict, df = convert_idx(df)
docu_udict, docu_idict = get_documents(df)
docu_udict_list=generate_vectors(bert_model_path,docu_udict)
docu_idict_list=generate_vectors(bert_model_path,docu_idict)
UserUserMatrix = np.zeros((user_num,user_num))
ItemItemMatrix = np.zeros((item_num,item_num))
UserUserMatrix = Matrix_sim(docu_udict_list,UserUserMatrix)
ItemItemMatrix = Matrix_sim(docu_idict_list,ItemItemMatrix)
MI_data=save_data(dataPath,df)

output = open(dataPath+'data.pkl', 'wb')
pickle.dump({ 'UserUserMatrix':UserUserMatrix,
             'ItemItemMatrix':ItemItemMatrix,'user_num':user_num, 'item_num':item_num}, output,protocol = 4)
output.close()




