#!/usr/bin/python

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import numpy as np

def parse_text(text):
    """ 
    1. tokenize
    2. convert to lowercase
    3. remove stopwords
    4. return list of words
    """

    stops = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = map(lambda f: f.lower(), tokens)
    tokens = filter(lambda f: f not in stops, tokens)
    return tokens


def parse_input_file(input_dir):
    """ Parses 'contexts.xml' and 'substitutions' files.
    
    Takes the directory of input files the only argument.
    
    Returns list of tasks in form of pairs (sentence, list of possible subs).
    Each sentence is a pair (list of strings, 0-based index of a string to be replaced).
    """

    tree = ET.parse(input_dir + 'contexts.xml')
    root = tree.getroot()
    
    ret = []
    for x in root.iter(tag='instance'):
        text = [c.strip() for c in x.itertext() if c.strip()]
        
        if len(text) == 3:
            pre = parse_text(text[0])
            sub = [' '.join(parse_text(text[1]))]
            suf = parse_text(text[2])
            ret.append((pre + sub + suf, len(pre)))
        else:
            if len(text[0]) < len(text[1]):
                sub = [' '.join(parse_text(text[0]))]
                suf = parse_text(text[1])
                ret.append((sub + suf, 0))
            else:
                pre = parse_text(text[0])
                sub = [' '.join(parse_text(text[1]))]
                ret.append((pre + sub, len(pre)))

    subs = open(input_dir + 'substitutions').read()
    subs = subs.split('\n')
    if subs[-1] == '':
        subs.pop()
    assert len(subs) == len(ret)

    assert all([x[-1] == ';' for x in subs])
    subs = map(lambda x: x.split(';')[:-1], subs)
    
    return zip(ret, subs)


def output(filepath, rankings):
    """ Writes rankings to file """
    
    f = open(filepath, 'w')
    for idx, ranking in enumerate(rankings):
        out = map(lambda x: '{' + x + '}', ranking)
        out = ' '.join(out)
        f.write('Sentence {} rankings: {}\n'.format(idx + 1, out))


def read_freqs_file(filepath):
    """ Read word frequencies file and returns dictionary
    where words are keys and frequencies are values.
    """
    f = open(filepath, 'r')
    ret = {}
    for line in f.read().split('\n'):
        tokens = line.split(' ')
        if len(tokens) == 2:
            ret[tokens[0]] = int(tokens[1])
    return ret
    

def lemmatizate_freqs_file(inp_file, out_file):
    lemmatizater = WordNetLemmatizer()
    freqs = read_freqs_file(inp_file)

    nfreqs = {}
    for word in freqs:
        nword = lemmatizater.lemmatize(word)
        if nword not in nfreqs:
            nfreqs[nword] = 0
        nfreqs[nword] += freqs[word]

    f = open(out_file, 'w')
    for word in sorted(nfreqs.keys()):
        f.write('{} {}\n'.format(word, nfreqs[word]))
    
