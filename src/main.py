#!/usr/bin/python

import utils
import gensim
import numpy as np
from nltk.corpus import wordnet
from scorer import Scorer

Word2Vec = gensim.models.Word2Vec

input_dir = '../data/trial-dataset/'
output_file = 'main_ranking'
simple_wiki_freqs_file = '../simple_wiki/freqs.txt'
wiki_freqs_file = '../wiki/freqs.txt'
model_file = '../models/glove.6B.200d.txt'
svm_file = 'train.dat'

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
            return 1.0 / (self.w_freqs[sub] / (self.sw_freqs[sub] * 1.0))
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
        self.model = Word2Vec.load_word2vec_format(model_file, binary=False)

    def similarity(self, a, b):
        if a in self.model and b in self.model:
            return self.model.similarity(a, b)
        return 0
    
    def score(self, (sentence, idx), sub):
        ret = 0
        for i, w in enumerate(sentence):
            ret += self.similarity(w, sub)
        return ret


class ScorerWordNet(Scorer):
    def score(self, _, sub):
        return len(wordnet.synsets(sub))


scorers = [
   ScorerInvLength(),
   ScorerWordNet(),
   ScorerSWFreqs(),
   ScorerWFreqs(),
#   ScorerContextSimilarity(),
]

def features((sentence, idx), sub):
    return map(lambda scorer: scorer.score((sentence, idx), sub), scorers)

gold_file = input_dir + 'substitutions.gold-rankings'
gold_rankings = utils.parse_rankings_file(gold_file)

utils.output_svm_file(svm_file, gold_rankings, tasks, features)

