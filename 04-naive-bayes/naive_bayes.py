from functools import reduce
import numpy as np
from math import log

def create_vocab_list(dataset):
    if (len(dataset) == 0): return []
    vocabset = set(reduce(lambda x, y: x + y, dataset))
    return list(vocabset)

def word2vec(vocab_list, words):
    return [1 if vocab in words else 0 for vocab in vocab_list]

def create_train_matrix(vocab_list, X):
    return [word2vec(vocab_list, feat) for feat in X]

def train_naive_bayes(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    
    # prefix p for probabilities
    p_abusive = np.sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words) # np.zeros(num_words)
    p1_num = np.ones(num_words) # np.zeros(num_words)
    
    p0_den = 2.0 # 0.0
    p1_den = 2.0 # 0.0
    
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_den += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_den += sum(train_matrix[i])
    p0_vect = np.log(p0_num / p0_den)
    p1_vect = np.log(p1_num / p1_den)
    return p0_vect, p1_vect, p_abusive

def classify_naive_bayes(vec2_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2_classify * p1_vec) + log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0