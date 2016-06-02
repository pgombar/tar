#!/usr/bin/python

import xml.etree.ElementTree as ET


def parse_input_file(input_dir):
    """ Parses 'contexts.xml' and 'substitutions' files.
    
    Takes the directory of input files the only argument.
    
    Returns list of tasks in form of pairs (sentence, list of possible subs).
    Each sentence is a pair (list of strings, 0-based index of a string to be replaced).

    e.g. of a task:
    ((['That boy is', 'bright', 'and young'], 1), ['intelligent', 'bright', 'clever', 'smart'])
    """

    tree = ET.parse(input_dir + 'contexts.xml')
    root = tree.getroot()

    ret = []
    for x in root.iter(tag='instance'):
        text = [c.strip() for c in x.itertext() if c.strip()]
        
        if len(text) == 3:
            ret.append((text, 1))
        else:
            if len(ret[0]) < len(ret[1]):
                ret.append((text, 0))
            else:
                ret.append((text, 1))

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
