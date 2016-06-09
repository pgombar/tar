#!/usr/bin/python

input_dir = '../data/test-data/'
output_file = 'rankings'

from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
            return math.log(self.freqs[sub])
        return 0        
    

class ScorerSWFreqs(Scorer):
    freqs = {}
    def __init__(self):
        self.freqs = utils.read_freqs_file(simple_wiki_freqs_file)
    def score(self, _, sub):
        if sub in self.freqs:
            return self.freqs[sub]
        return 0


class ScorerFreqsRatio(Scorer):
    s1 = 0
    s2 = 0
    
    def __init__(self):
        self.s1 = ScorerSWFreqs()
        self.s2 = ScorerWFreqs()

    def score(self, _, sub):
        a = self.s1.score(_, sub)
        b = self.s2.score(_, sub)
        if b != 0:
            return a / b
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
gold_rankings = utils.parse_rankings_file(gold_file)

scorers = [
    ScorerInvLength(),
#    ScorerWordNet(),
#    ScorerSWFreqs(),
#    ScorerWFreqs(),
#    ScorerFreqsRatio(),
    ScorerContextSimilarity(),
#    ScorerSemanticSimilarity(),
]

def get_features(tasks):
    features = []
    for (sentence, idx), subs in tasks:
        for sub in subs:
            features.append(map(lambda scorer: scorer.score((sentence, idx), sub), scorers))
    return features

print 'initialization done'

coefs = np.zeros(len(scorers))
features = get_features(tasks)
# transformer = MinMaxScaler().fit(features)
# features = transformer.transform(features)

best = 0
best_coefs = []

K = 10

def rec(i):
    if i == len(scorers):
        global features
        rankings = utils.recover_rankings(tasks, -np.dot(features, coefs))
        score = utils.evaluate(rankings, gold_rankings)
        global best, best_coefs
        if score > best:
            print coefs, score
            best = score
            best_coefs = np.array(coefs, copy=True)
        return

    coefs[i] = 0
    while coefs[i] <= K:
        rec(i + 1)
        coefs[i] += 1

rec(0)
print best
print best_coefs
    
test_dir = '../data/test-data/'
test_tasks = utils.parse_input_file(test_dir)
test_gold_file = test_dir + 'substitutions.gold-rankings'
test_gold_rankings = utils.parse_rankings_file(test_gold_file)

features = get_features(test_tasks)
#features = transformer.transform(features)

rankings = utils.recover_rankings(test_tasks, -np.dot(features, coefs))
print utils.evaluate(rankings, test_gold_rankings)
