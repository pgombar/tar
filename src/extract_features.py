#!/usr/bin/python

import argparse

parser = argparse.ArgumentParser(
    description='Features extractor.',
    epilog='Example: ./extract_features.py ../data/trial-dataset/ train.dat')

parser.add_argument('input_dir', help='Samples directory')
parser.add_argument('output_file', help='Output file')

args = parser.parse_args()
input_dir = args.input_dir
output_file = args.output_file



import utils
import gensim
import numpy as np
from nltk.corpus import wordnet
from scorer import Scorer

Word2Vec = gensim.models.Word2Vec

simple_wiki_freqs_file = '../simple_wiki/freqs.txt'
wiki_freqs_file = '../wiki/freqs.txt'
w2v_model_file = '../models/glove.6B.200d.txt'

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
        self.model = Word2Vec.load_word2vec_format(w2v_model_file, binary=False)

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
utils.output_features_file(output_file, gold_rankings, tasks, features)

