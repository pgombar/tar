#!/usr/bin/python

import itertools
import numpy as np
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model, cross_validation
from sklearn.preprocessing import PolynomialFeatures

input_dir = '../data/trial-dataset/'
output_file = 'rankings'

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer

import utils
import math
import numpy as np
from scorer import *

tasks = utils.parse_input_file(input_dir)

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

best_coef = []
best_score = 0
best_scaler = ''
best_scaler_name = ''
best_deg = 0
best_C = 0

tasks_train, tasks_test, gold_train, gold_test = cross_validation.train_test_split(
    tasks, gold, train_size=0.7, random_state=42)

X, y, blocks = get_features(tasks, gold)

scalers = {'standard': StandardScaler,
           'minmax': MinMaxScaler,
           #'none': Imputer,
}

for deg in (1,2):
    for scaler_name in scalers.keys():
        scaler = scalers[scaler_name]()
    
        X_train, y_train, blocks_train = get_features(tasks_train, gold_train)
        X_test, y_test, blocks_test = get_features(tasks_test, gold_test)
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = PolynomialFeatures(degree=deg).fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_test = PolynomialFeatures(degree=deg).fit_transform(X_test)

        for logC in range(-15, 8):
            C = 2**logC
            comb = itertools.combinations(range(X_train.shape[0]), 2)
            k = 0
            Xp, yp, diff = [], [], []
            for (i, j) in comb:
                if y_train[i] == y_train[j] or blocks_train[i] != blocks_train[j]:
                    continue
                Xp.append(X_train[i] - X_train[j])
                diff.append(y_train[i] - y_train[j])
                yp.append(np.sign(diff[-1]))

            clf = svm.SVC(kernel='linear', C=C, max_iter=100000000)
            clf.fit(Xp, yp)
            coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)

            rankings = utils.recover_rankings(tasks_test, np.dot(X_test, coef))
            score = utils.evaluate(rankings, gold_test)
            
            print 'scaler={}, deg={}, C={}: score={}'.format(scaler_name, deg, C, score)
            if score > best_score:
                best_score = score
                best_C = C
                best_coef = np.copy(coef)
                best_scaler = scaler
                best_scaler_name = scaler_name
                best_deg = deg
            
print '\n\n\ndone'
print 'scaler: ', best_scaler_name
print 'deg: ', best_deg
print 'C: ', best_C
print 'coef: ', best_coef

test_dir = '../data/test-data/'
test_tasks = utils.parse_input_file(test_dir)
test_gold_file = test_dir + 'substitutions.gold-rankings'
test_gold_rankings = utils.parse_rankings_file(test_gold_file)

testX, testy, testblocks = get_features(test_tasks, test_gold_rankings)
testX = best_scaler.transform(testX)
testX = PolynomialFeatures(degree=best_deg).fit_transform(testX)

rankings = utils.recover_rankings(test_tasks, np.dot(testX, best_coef))
print 'test dataset: ', utils.evaluate(rankings, test_gold_rankings)
utils.output('trainer_rankings', rankings)
