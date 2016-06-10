#!/usr/bin/python

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


scorers = [
    ScorerInvLength(),
    ScorerWordNet(),
    ScorerSWFreqs(),
    ScorerFreqsRatio(),
    ScorerContextSimilarity(),
]

def get_features(tasks):
    features = []
    for (sentence, idx), subs in tasks:
        for sub in subs:
            features.append(map(lambda scorer: scorer.score((sentence, idx), sub), scorers))
    return features

print 'initialization done'

coefs = [1,1,1,1,1]
    
test_dir = '../data/test-data/'
test_tasks = utils.parse_input_file(test_dir)
test_gold_file = test_dir + 'substitutions.gold-rankings'
test_gold_rankings = utils.parse_rankings_file(test_gold_file)

features = get_features(test_tasks)
transformer = MinMaxScaler().fit(features)
features = transformer.transform(features)

rankings = utils.recover_rankings(test_tasks, -np.dot(features, coefs))
print utils.evaluate(rankings, test_gold_rankings)
utils.output('unsupervised_rankings', rankings)
