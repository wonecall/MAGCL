'''
Created on Oct 1, 2022
Tensorflow Implementation of Multi-Aspect Graph Contrastive Learning (MAGCL) model 
@author: Ke Wang (onecall@sjtu.edu.cn)
version:
evaluation for top-k recommendation in util folder
'''

from pandas import Series, DataFrame
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import random
from itertools import islice
import tensorflow as tf
import argparse
import scipy.sparse as sp
import os, sys, time, math
from tqdm import tqdm
sys.path.append("..")
from util import metrics
sys.path.append("..")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_device', type=int, default=0,
                    help='choose which gpu to run')
parser.add_argument('--batch_size', type=int, default=100,
                    help='size of mini-batch')
parser.add_argument('--train_neg_num', type=int, default=4,
                    help='number of negative samples per training positive sample')
parser.add_argument('--test_size', type=int, default=1,
                    help='size of sampled test data')
parser.add_argument('--test_neg_num', type=int, default=99,
                    help='number of negative samples for test')
parser.add_argument('--epochs', type=int, default=150,
                    help='the number of epochs')
parser.add_argument('--gnn_layers', nargs='?', default=[64,64,64],
                    help='the unit list of layers')
parser.add_argument('--mlp_layers', nargs='?', default=[32,16,8],
                    help='the unit list of layers')
parser.add_argument('--embedding_size', type=int, default=64,
                    help='the size for embedding user and item')
parser.add_argument('--topK', type=int, default=10,
                    help='topk for evaluation')
parser.add_argument('--regularizer_rate', type=float, default=8e-5,   
                    help='the regularizer rate')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--dropout_message', type=float, default=0, 
                    help='dropout rate of message')
parser.add_argument('--NCForMF', type=str, default='MF',
                    help='method to propagate embeddings')
args = parser.parse_args(args=[])
print(args)


def _load_num(df):
    uuu=0
    iii=0
    for user, item in zip(df['uid'], df['iid']):
        if uuu < user:
            uuu = user
        if iii < item:
            iii = item        
    return uuu+1,iii+1

def _construct_pos_dict(df):
    pos_dict = defaultdict(set)
    for user, item in zip(df['uid'], df['iid']):
        pos_dict[user].add(item)
    return pos_dict

def read_data(dataPath):
    inFile = open(dataPath + 'data.pkl','rb')
    data = pickle.load(inFile)
    UserUserMatrix,ItemItemMatrix = data['UserUserMatrix'], data['ItemItemMatrix']
    inFile.close()
    return UserUserMatrix,ItemItemMatrix

def split_dataset(dataset,train_ratio=0.8):
    indices = list(range(len(dataset)))  
    train_len = int(len(dataset) * train_ratio)
    train_indices = random.sample(indices, train_len)   
    train_set = [dataset[i] for i in train_indices]  
    test_set = [dataset[j] for j in indices if j not in train_indices]  
    return train_set, test_set

def load_dataset(file_path):
    inFile_s = open(file_path)
    data_set = []
    for line in islice(inFile_s, 1, None):
        line = line.strip().split(',')
        userIndex = int(line[4])
        itemIndex = int(line[5])
        data_set.append((userIndex, itemIndex))
    return split_dataset(data_set)

def _add_negtive(user, item, num_items, pos_dict, neg_num, boolindex):
    user, item, num_items, pos_dict, neg_num, train = user, item, num_items, pos_dict,neg_num ,boolindex
    users, items, labels = [], [], []
    neg_set = set(range(num_items)).difference(pos_dict[user])  
    neg_sample_list = np.random.choice(list(neg_set), neg_num, replace=False).tolist()
    for neg_sample in neg_sample_list:
        users.append(user)
        items.append(neg_sample)
        labels.append(0) if train == True else labels.append(neg_sample)
    users.append(user)
    items.append(item)
    if train == True:
        labels.append(1)
    else:
        labels.append(int(item))
    return (users, items, labels)


class MAGCL(object):
    def __init__(self, args, iterator, norm_adj_mat1, norm_adj_mat2, num_users, num_items,is_training):
        self.args = args
        self.iterator = iterator
        self.norm_adj_mat1 = norm_adj_mat1
        self.norm_adj_mat2 = norm_adj_mat2
        self.num_users = num_users
        self.num_items = num_items
        self.is_training = is_training
        self.n_fold = 50
        self.get_data()
        self.all_weights = self.init_weights()
        self.item_embeddings_s1, self.user_embeddings1 = self.create_lightgcn_embed1()
        self.item_embeddings_s21, self.user_embeddings21 = self.create_lightgcn_embed21()
        self.item_embeddings_s22, self.user_embeddings22 = self.create_lightgcn_embed22()    
        self.item_embeddings_s23, self.user_embeddings23 = self.create_lightgcn_embed23()
        self.item_embeddings_s24, self.user_embeddings24 = self.create_lightgcn_embed24()    
        self.item_embeddings_s25, self.user_embeddings25 = self.create_lightgcn_embed25() 
        self.item_embeddings_s = self.item_embeddings_s1 + 0.10 * self.item_embeddings_s21\
                                 +0.10*self.item_embeddings_s22+0.10*self.item_embeddings_s23\
                                 +0.10*self.item_embeddings_s24+0.10*self.item_embeddings_s25
        self.user_embeddings = self.user_embeddings1 + 0.10 * self.user_embeddings21\
                                + 0.10 * self.user_embeddings22+0.10 * self.user_embeddings23\
                                + 0.10 * self.user_embeddings24+0.10 * self.user_embeddings25
        self.item_embeddings_s1_ssl, self.user_embeddings1_ssl = self.creat_gcn_embedd1_ssl()
        self.item_embeddings_s21_ssl =  self.item_embeddings_s21+self.item_embeddings_s22+self.item_embeddings_s23\
                                        + self.item_embeddings_s24 + self.item_embeddings_s25
        self.user_embeddings21_ssl =  self.user_embeddings21+self.user_embeddings22+self.user_embeddings23\
                                        + self.user_embeddings24 + self.user_embeddings25
        self.inference()
        self.saver = tf.train.Saver(tf.global_variables())
    def get_data(self):
        sample = self.iterator.get_next()
        self.user, self.item_s= sample['user'], sample['item']
        self.label_s = tf.cast(sample['label'], tf.float32)
    def init_weights(self):
        all_weights = dict()
        initializer = tf.truncated_normal_initializer(0.01)
        regularizer = tf.contrib.layers.l2_regularizer(self.args.regularizer_rate) 
        all_weights['user_embeddings1'] = tf.get_variable(
            'user_embeddings1', (self.num_users, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_s1'] = tf.get_variable(
            'item_embeddings_s1', (self.num_items, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['user_embeddings21'] = tf.get_variable(
            'user_embeddings21', (self.num_users, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_s21'] = tf.get_variable(
            'item_embeddings_s21', (self.num_items, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['user_embeddings22'] = tf.get_variable(
            'user_embeddings22', (self.num_users, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_s22'] = tf.get_variable(
            'item_embeddings_s22', (self.num_items, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['user_embeddings23'] = tf.get_variable(
            'user_embeddings23', (self.num_users, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_s23'] = tf.get_variable(
            'item_embeddings_s23', (self.num_items, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['user_embeddings24'] = tf.get_variable(
            'user_embeddings24', (self.num_users, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_s24'] = tf.get_variable(
            'item_embeddings_s24', (self.num_items, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['user_embeddings25'] = tf.get_variable(
            'user_embeddings25', (self.num_users, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_s25'] = tf.get_variable(
            'item_embeddings_s25', (self.num_items, self.args.embedding_size), tf.float32, initializer, regularizer)          
        self.layers_plus = [self.args.embedding_size] + self.args.gnn_layers
        print("self.layers_plus",self.layers_plus)
        for k in range(len(self.layers_plus)-1):
            all_weights['W_ssl_%d' % k] = tf.get_variable(
                'W_ssl_%d'% k, (self.layers_plus[k], self.layers_plus[k+ 1]), tf.float32, initializer, regularizer)
            all_weights['b_ssl_%d' % k] = tf.get_variable(
                'b_ssl_%d'% k, self.layers_plus[k+ 1], tf.float32, tf.zeros_initializer(), regularizer)                       
        return all_weights 
    ########################GCN###############################################    
    def creat_gcn_embedd1_ssl(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat1)
        embeddings = tf.concat([self.all_weights['item_embeddings_s1'], self.all_weights['user_embeddings1']], axis=0)
        all_embeddings = [embeddings]
        for k in range(len(self.layers_plus)-1):
            temp_embedd = [tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings) for f in range(self.n_fold)]
            embeddings = tf.concat(temp_embedd, axis=0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.all_weights['W_ssl_%d'%k])
                                          + self.all_weights['b_ssl_%d'%k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.args.dropout_message)
            all_embeddings += [embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        item_embeddings_s, user_embeddings = tf.split(all_embeddings, [self.num_items, self.num_users], axis=0)
        return item_embeddings_s, user_embeddings
    ########################lightGCN###############################################
    def create_lightgcn_embed1(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat1)  
        ego_embeddings = tf.concat([self.all_weights['item_embeddings_s1'], self.all_weights['user_embeddings1']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers_plus)-1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        item_embeddings_s, user_embeddings = tf.split(all_embeddings, [self.num_items, self.num_users], axis=0)
        return item_embeddings_s, user_embeddings
    def create_lightgcn_embed21(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat2)  
        ego_embeddings = tf.concat([self.all_weights['item_embeddings_s21'], self.all_weights['user_embeddings21']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers_plus)-1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        item_embeddings_s, user_embeddings = tf.split(all_embeddings, [self.num_items, self.num_users], axis=0)
        return item_embeddings_s, user_embeddings
    def create_lightgcn_embed22(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat2)  
        ego_embeddings = tf.concat([self.all_weights['item_embeddings_s22'], self.all_weights['user_embeddings22']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers_plus)-1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        item_embeddings_s, user_embeddings = tf.split(all_embeddings, [self.num_items, self.num_users], axis=0)
        return item_embeddings_s, user_embeddings
    def create_lightgcn_embed23(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat2)  
        ego_embeddings = tf.concat([self.all_weights['item_embeddings_s23'], self.all_weights['user_embeddings23']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers_plus)-1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        item_embeddings_s, user_embeddings = tf.split(all_embeddings, [self.num_items, self.num_users], axis=0)
        return item_embeddings_s, user_embeddings
    def create_lightgcn_embed24(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat2)  
        ego_embeddings = tf.concat([self.all_weights['item_embeddings_s24'], self.all_weights['user_embeddings24']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers_plus)-1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        item_embeddings_s, user_embeddings = tf.split(all_embeddings, [self.num_items, self.num_users], axis=0)
        return item_embeddings_s, user_embeddings
    def create_lightgcn_embed25(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat2)  
        ego_embeddings = tf.concat([self.all_weights['item_embeddings_s25'], self.all_weights['user_embeddings25']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers_plus)-1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        item_embeddings_s, user_embeddings = tf.split(all_embeddings, [self.num_items, self.num_users], axis=0)
        return item_embeddings_s, user_embeddings
    ################################################################################################# 
    def _split_A_hat(self, X):
        fold_len = math.ceil((X.shape[0]) / self.n_fold)
        A_fold_hat = [self._convert_sp_mat_to_sp_tensor( X[i_fold*fold_len :(i_fold+1)*fold_len])
                      for i_fold in range(self.n_fold)]
        return A_fold_hat
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    def inference(self):      
        initializer = tf.truncated_normal_initializer(0.01)
        regularizer = tf.contrib.layers.l2_regularizer(self.args.regularizer_rate)
        with tf.name_scope('embedding'):
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user)
            item_embedding_s = tf.nn.embedding_lookup(self.item_embeddings_s, self.item_s)          
            item_embeddings_s1_ssl = tf.nn.embedding_lookup(self.item_embeddings_s1_ssl, self.item_s)
            user_embeddings1_ssl = tf.nn.embedding_lookup(self.user_embeddings1_ssl, self.user)
            item_embeddings_s21_ssl = tf.nn.embedding_lookup(self.item_embeddings_s21_ssl, self.item_s)
            user_embeddings21_ssl = tf.nn.embedding_lookup(self.user_embeddings21_ssl, self.user)
            emb_merge1 = tf.concat([item_embeddings_s1_ssl, user_embeddings1_ssl], axis=0)
            emb_merge2 = tf.concat([item_embeddings_s21_ssl, user_embeddings21_ssl], axis=0)
        with tf.name_scope('propagation'):
            if self.args.NCForMF == 'MF':
                self.logits_dense_s = tf.reduce_sum(tf.multiply(user_embedding, item_embedding_s), 1) 
            elif self.args.NCForMF == 'NCF':
                a_s = tf.concat([user_embedding, item_embedding_s], axis=-1, name='inputs_s')
                for i, units in enumerate(self.args.mlp_layers):
                    dense_s = tf.layers.dense(a_s, units, tf.nn.relu, kernel_initializer=initializer,
                                          kernel_regularizer = regularizer, name='dense_s_%d' % i)
                    a_s = tf.layers.dropout(dense_s, self.args.dropout_message)
                self.logits_dense_s = tf.layers.dense(inputs=a_s,
                                                      units=1,
                                                      kernel_initializer=initializer,
                                                      kernel_regularizer=regularizer,
                                                      name='logits_dense_s')
            else:
                raise ValueError
            self.logits_s = tf.squeeze(self.logits_dense_s)

            loss_list_s = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_s, logits=self.logits_s,     
                                                                  name='loss_s')
            loss_w_s = tf.map_fn(lambda x: tf.cond(tf.equal(x, 1.0), lambda: 5.0, lambda: 1.0), self.label_s)

            self.loss_s = tf.reduce_mean(tf.multiply(loss_list_s, loss_w_s))
            normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
            normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)
            pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
            ttl_score = tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_a=False, transpose_b=True)
            pos_score = tf.exp(pos_score / 5)
            ttl_score = tf.reduce_sum(tf.exp(ttl_score / 5), axis=1)
            ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
            self.ssl_loss = 0.05 * ssl_loss            
            self.loss = self.loss_s + self.ssl_loss
            self.optimizer = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)
            self.label_replica_s = self.label_s
            _, self.indice_s = tf.nn.top_k(tf.sigmoid(self.logits_s), self.args.topK)
    def step(self, sess):
        if self.is_training:
            label_s, indice_s, loss, optim = sess.run(
                [self.label_replica_s, self.indice_s, self.loss, self.optimizer])
            return loss
        else:
            label_s, indice_s = sess.run([self.label_replica_s, self.indice_s])
            prediction_s = np.take(label_s, indice_s)
            return prediction_s, label_s

def evaluate(predictions, labels):
    label = int(labels[-1])
    hr = metrics.hit(label, predictions)
    mrr = metrics.mrr(label, predictions)
    ndcg = metrics.ndcg(label, predictions)
    return hr, mrr, ndcg

def normalized_adj_single(adj):
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj)
    return norm_adj

def load_mat_s1(num_users, num_items, data_dict_str,UserUserMatrix,ItemItemMatrix, args):
    num_users = num_users
    num_items = num_items
    train_df = {'user':data_dict_str['user'][args.train_neg_num::args.train_neg_num+1],
                  'item':data_dict_str['item'][args.train_neg_num::args.train_neg_num+1]} 
    R_s = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    U_s = sp.dok_matrix(UserUserMatrix, dtype=np.float32)
    I_s = sp.dok_matrix(ItemItemMatrix, dtype=np.float32)
    for user, item in zip(train_df['user'], train_df['item']):
        R_s[int(user), int(item)] = 1.0
    plain_adj_mat = sp.dok_matrix((num_items+ num_users, num_items+ num_users),
                                  dtype=np.float32).tolil()
    plain_adj_mat[num_items: num_items+ num_users, :num_items] = R_s
    plain_adj_mat[:num_items, num_items: num_items+ num_users] = R_s.T
    plain_adj_mat[:num_items, :num_items] =  0.0*I_s
    plain_adj_mat[num_items: num_items+ num_users, num_items: num_items+ num_users] =  0.0*U_s
    plain_adj_mat = plain_adj_mat.todok()
    norm_adj_mat_s1 = normalized_adj_single(plain_adj_mat+ sp.eye(plain_adj_mat.shape[0]))
    return norm_adj_mat_s1

def load_mat_s2(num_users, num_items, data_dict_str,UserUserMatrix,ItemItemMatrix, args):
    num_users = num_users
    num_items = num_items
    train_df = {'user':data_dict_str['user'][args.train_neg_num::args.train_neg_num+1],
                  'item':data_dict_str['item'][args.train_neg_num::args.train_neg_num+1]} 
    R_s = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    U_s = sp.dok_matrix(UserUserMatrix, dtype=np.float32)
    I_s = sp.dok_matrix(ItemItemMatrix, dtype=np.float32)
    for user, item in zip(train_df['user'], train_df['item']):
        R_s[int(user), int(item)] = 1.0
    plain_adj_mat = sp.dok_matrix((num_items+ num_users, num_items+ num_users),
                                  dtype=np.float32).tolil()
    plain_adj_mat[num_items: num_items+ num_users, :num_items] = 0.0*R_s
    plain_adj_mat[:num_items, num_items: num_items+ num_users] = 0.0*R_s.T
    plain_adj_mat[:num_items, :num_items] =  0.1*I_s
    plain_adj_mat[num_items: num_items+ num_users, num_items: num_items+ num_users] =  0.1*U_s
    plain_adj_mat = plain_adj_mat.todok()
    norm_adj_mat_s2 = normalized_adj_single(plain_adj_mat+ sp.eye(plain_adj_mat.shape[0]))
    return norm_adj_mat_s2


if __name__ == '__main__':
    with tf.Session() as sess:
        file_path = './data/Digital_Music.csv'
        dataPath = './data/'
        name_s = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path, sep=',')
        num_users, num_items = _load_num(df)
        pos_dict=_construct_pos_dict(df)
        trainData, testData = load_dataset(file_path)
        UserUserMatrix,ItemItemMatrix = read_data(dataPath)

        train_df1,train_df2 = [],[]
        test_df1,test_df2 = [],[]
        train_df,test_df = {'uid':[],'iid':[]},{'uid':[],'iid':[]}
        train_df,test_df =  pd.DataFrame(train_df),pd.DataFrame(test_df)
        for u,i in trainData:   
            train_df1.append(u)
            train_df2.append(i)
        train_df['uid'] = train_df1
        train_df['iid'] = train_df2 
        for u,i in testData:
            test_df1.append(u)
            test_df2.append(i)
        test_df['uid'] = test_df1
        test_df['iid'] = test_df2 

        users = []
        items = []
        labels = []    
        for user, item in zip(train_df['uid'], train_df['iid']):
            batch_users, batch_items, batch_labels = _add_negtive(user, item, num_items, pos_dict,4 ,True)
            users += batch_users
            items += batch_items
            labels += batch_labels
        users = list(map(int, users))
        items = list(map(int, items))
        labels = list(map(int, labels))
        data_dict_str = {'user': users, 'item': items, 'label': labels}
        users = []
        items = []
        labels = []    
        for user, item in zip(test_df['uid'], test_df['iid']):
            batch_users, batch_items, batch_labels = _add_negtive(user, item, num_items, pos_dict,99 ,False)
            users += batch_users
            items += batch_items
            labels += batch_labels
        users = list(map(int, users))
        items = list(map(int, items))
        labels = list(map(int, labels))
        data_dict_ste = {'user': users, 'item': items, 'label': labels}

        norm_adj_mat1 = load_mat_s1(num_users, num_items, data_dict_str,UserUserMatrix,ItemItemMatrix, args)
        norm_adj_mat2 = load_mat_s2(num_users, num_items, data_dict_str,UserUserMatrix,ItemItemMatrix, args)
        train_data = tf.data.Dataset.from_tensor_slices(data_dict_str)
        train_data = train_data.shuffle(buffer_size=len(data_dict_str['user'])).batch(args.batch_size)
        print('train_data',train_data)
        print('train_data/batch_size',len(data_dict_str['user'])/args.batch_size)
        test_data = tf.data.Dataset.from_tensor_slices(data_dict_ste)
        print('test_data',test_data) 
        test_data = test_data.batch(args.test_size + args.test_neg_num)

        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        model = MAGCL(args, iterator, norm_adj_mat1, norm_adj_mat2, num_users, num_items, True)
        print("Creating model with original parameters...")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()    
        count = 0
        loss = 0
        last_count = 0
        hr_s_list, mrr_s_list, ndcg_s_list = [], [], []
        for epoch in range(1, args.epochs + 1):
            print('=' * 30 + ' EPOCH %d ' % epoch + '=' * 30)
            ################################## Training ################################
            if 6 > epoch > 3:
                model.args.lr = 1e-3
            if epoch >= 6:
                model.args.lr = 1e-4
            sess.run(model.iterator.make_initializer(train_data))
            model.is_training = True
            start_time = time.time()
            try:
                while True:
                    count += 1
                    loss += model.step(sess)
                    if count % 300 == 0:
                        print('Epoch %d, step %d, with average loss of %.4f in last %d steps;'
                              % (epoch, count, loss / (count - last_count), count - last_count))
                        loss = 0
                        last_count = count
            except tf.errors.OutOfRangeError:
                print("Epoch %d, finish training " % epoch + "took " +
                      time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)) + ';')
            ################################## Testing ################################
            sess.run(model.iterator.make_initializer(test_data))
            model.is_training = False
            start_time = time.time()
            HR_s, MRR_s, NDCG_s = [], [], []
            predictions_s, labels_s = model.step(sess)
            cnt = 1
            try:
                while True:
                    predictions_s, labels_s= model.step(sess)
                    hr_s, mrr_s, ndcg_s = evaluate(predictions_s, labels_s)
                    HR_s.append(hr_s)
                    MRR_s.append(mrr_s)
                    NDCG_s.append(ndcg_s)
                    cnt += 1
            except tf.errors.OutOfRangeError:
                hr_s = np.array(HR_s).mean()
                mrr_s = np.array(MRR_s).mean()
                ndcg_s = np.array(NDCG_s).mean()
                hr_s_list.append(hr_s)
                mrr_s_list.append(mrr_s)
                ndcg_s_list.append(ndcg_s)
                print("Epoch %d, finish testing " % epoch + "took: " +
                      time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)) + ';')
                print('Epoch %d, %s HR is %.4f, MRR is %.4f, NDCG is %.4f;' %
                      (epoch, name_s, hr_s, mrr_s, ndcg_s))
            save_path=saver.save(sess, dataPath + "model.ckpt")
        print('=' * 30 + 'Finish training' + '=' * 30)
        print('%s best HR is %.4f, MRR is %.4f, NDCG is %.4f;' %
              (name_s, max(hr_s_list), max(mrr_s_list), max(ndcg_s_list)))



