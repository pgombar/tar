#!/usr/bin/python

import utils
import word2vec
import numpy as np

input_dir = '../data/trial-dataset/'
output_file = 'main_ranking'
simple_wiki_freqs_file = '../simple_wiki/freqs.txt'

tasks = utils.parse_input_file(input_dir)


def rank_stupid(s):
    ret = []
    for _, subs in s:
        ret.append(sorted(subs, key=len))
    return ret


def rank_by_freqs(s, freqs):
    def f(word):
        if word in freqs:
            return -freqs[word]
        return 0
    
    ret = []
    for _, subs in s:
        ret.append(sorted(subs, key=f))
    return ret


def rank_by_context_similarity(s):
    model = word2vec.load('../simple_wiki/dump.bin')
    ret = []
    for (sentence, idx), subs in s:
        word = sentence[idx]
        def score(sub):
            ret_score = 0
            for i, w in enumerate(sentence):
                if i != idx:
                    ret_score += utils.word2vec_similarity(model, w, sub)
            return -ret_score
        bla = map(lambda sub: (score(sub), sub), subs)
        bla = sorted(bla)
        ret.append(map(lambda (_, a): a, bla))
        ret.append(sorted(subs, key=score))
        print 'done'
    return ret
    
simple_wiki_freqs = utils.read_freqs_file(simple_wiki_freqs_file)
rankings = rank_by_context_similarity(tasks)
utils.output(output_file, rankings)

