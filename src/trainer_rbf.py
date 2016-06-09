#!/usr/bin/python

import itertools
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model, cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import rbf_kernel

input_dir = '../data/trial-dataset/'
output_file = 'rankings'

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer

import utils
import math
import gensim
import numpy as np
from nltk.corpus import wordnet
from scorer import Scorer

Word2Vec = gensim.models.Word2Vec

simple_wiki_freqs_file = '../simple_wiki/freqs.txt'
wiki_freqs_file = '../wiki/freqs.txt'
w2v_model_file = '../models/glove.6B.200d_reduced.txt'

tasks = utils.parse_input_file(input_dir)

class ScorerInvLength(Scorer):
    def score(self, _, sub):
        return 1.0 / len(sub)

class ScorerCorpusComplexity(Scorer):
    sw_freqs = {}
    w_freqs = {}
    def __init__(self):
        self.sw_freqs = utils.read_freqs_file(simple_wiki_freqs_file)
        self.w_freqs = utils.read_freqs_file(wiki_freqs_file, separator='\t')
    def score(self, _, sub):
        if sub in self.w_freqs and sub in self.sw_freqs:
            return self.sw_freqs[sub] / self.w_freqs[sub]
        return 0


class ScorerWFreqs(Scorer):
    freqs = {}
    def __init__(self):
        self.freqs = utils.read_freqs_file(wiki_freqs_file, separator='\t')
    def score(self, _, sub):
        if sub in self.freqs:
            return self.freqs[sub]
        return 0        
    

class ScorerSWFreqs(Scorer):
    freqs = {}
    def __init__(self):
        self.freqs = utils.read_freqs_file(simple_wiki_freqs_file)
    def score(self, _, sub):
        if sub in self.freqs:
            return self.freqs[sub]
        return 0

    
class ScorerContextSimilarity(Scorer):
    model = 0
    def __init__(self):
        self.model = Word2Vec.load_word2vec_format(w2v_model_file, binary=False)

    def similarity(self, a, b):
        if a in self.model and b in self.model:
            return self.model.similarity(a, b)
        return 0
    
    def score(self, (sentence, idx), sub):
        ret = 0
        for i, w in enumerate(sentence):
            if i != idx:
                ret += self.similarity(w, sub)
        return ret

    
class ScorerSemanticSimilarity(Scorer):
    model = 0
    def __init__(self):
        self.model = Word2Vec.load_word2vec_format(w2v_model_file, binary=False)

    def similarity(self, a, b):
        if a in self.model and b in self.model:
            return self.model.similarity(a, b)
        return 0
    
    def score(self, (sentence, idx), sub):
        return self.similarity(sentence[idx], sub)

class ScorerWordNet(Scorer):
    def score(self, _, sub):
        return len(wordnet.synsets(sub))


gold_file = input_dir + 'substitutions.gold-rankings'
gold = utils.parse_rankings_file(gold_file)



scorers = [
    ScorerInvLength(),
    ScorerWordNet(),
    ScorerSWFreqs(),
    ScorerWFreqs(),
    ScorerCorpusComplexity(),
    ScorerContextSimilarity(),
    ScorerSemanticSimilarity(),
 ]

def get_features(tasks, gold):
    X = []
    y = []
    blocks = []
    task_idx = 0
    for ((sentence, idx), subs), rankings in zip(tasks, gold):
        for sub in subs:
            X.append(map(lambda scorer: scorer.score((sentence, idx), sub), scorers))
            y.append(gold[task_idx][sub])
            blocks.append(task_idx)
        task_idx += 1
    return np.array(X), np.array(y), np.array(blocks)

best_score = 0
best_scaler = ''
best_scaler_name = ''
best_deg = 0
best_C = 0
best_model = 0
best_gamma = 0

tasks_train, tasks_test, gold_train, gold_test = cross_validation.train_test_split(
    tasks, gold, train_size=0.7, random_state=97)

scalers = {'standard': StandardScaler,
           #'minmax': MinMaxScaler,
           #'none': Imputer,
}

def predict(clf, gamma, X):
    return np.dot(clf.dual_coef_, rbf_kernel(clf.support_vectors_, X, gamma=gamma))[0]

for deg in [1]:
    for scaler_name in scalers.keys():
        scaler = scalers[scaler_name]()
    
        X_train, y_train, blocks_train = get_features(tasks_train, gold_train)
        X_test, y_test, blocks_test = get_features(tasks_test, gold_test)
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = PolynomialFeatures(degree=deg).fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_test = PolynomialFeatures(degree=deg).fit_transform(X_test)

        for logGamma in range(-10, 5):
            gamma = 2**logGamma
            for logC in range(-10, 5):
                C = 2**logC
                Xp, yp = [], []
                for i in range(X_train.shape[0]):
                    j = i
                    while j < X_train.shape[0] and blocks_train[j] == blocks_train[i]:
                        if y_train[i] != y_train[j]:
                            Xp.append(X_train[i] - X_train[j])
                            yp.append(np.sign(y_train[i] - y_train[j]))
                        j += 1
                        
                clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
                clf.fit(Xp, yp)

                rankings = utils.recover_rankings(tasks_test, predict(clf, gamma, X_test))
                score = utils.evaluate(rankings, gold_test)
                
                print 'scaler={}, deg={}, gamma={}, C={}: score={}'.format(scaler_name, deg, gamma, C, score)
                if score > best_score:
                    best_score = score
                    best_C = C
                    best_scaler = scaler
                    best_scaler_name = scaler_name
                    best_deg = deg
                    best_model = clf
                    best_gamma = gamma
                
print '\n\n\ndone'
print 'scaler: ', best_scaler_name
print 'deg: ', best_deg
print 'C: ', best_C
print 'gamma: ', best_gamma

test_dir = '../data/test-data/'
test_tasks = utils.parse_input_file(test_dir)
test_gold_file = test_dir + 'substitutions.gold-rankings'
test_gold_rankings = utils.parse_rankings_file(test_gold_file)

testX, testy, testblocks = get_features(test_tasks, test_gold_rankings)
testX = best_scaler.transform(testX)
testX = PolynomialFeatures(degree=best_deg).fit_transform(testX)

rankings = utils.recover_rankings(test_tasks, predict(best_model, best_gamma, testX))

print 'test dataset: ', utils.evaluate(rankings, test_gold_rankings)
utils.output('rankings', rankings)
