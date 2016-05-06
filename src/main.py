#!/usr/bin/python

import xml.etree.ElementTree as ET

testdir = '../data/trial-dataset/'

def parse_input():
    tree = ET.parse(testdir + 'contexts.xml')
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

    subs = open(testdir + 'substitutions').read()
    subs = subs.split('\n')
    if subs[-1] == '':
        subs.pop()
    assert len(subs) == len(ret)

    assert all([x[-1] == ';' for x in subs])
    subs = map(lambda x: x.split(';')[:-1], subs)
    
    return zip(ret, subs)

def solve(s):
    ret = []
    for _, subs in s:
        ret.append(sorted(subs, key=len))
    return ret



s = parse_input()

# s -> list of input sentences
# s[i] = ((list of strings, idx of string to be replaced), list of possible substitutions)
# e.g. s[i] = ((['During the siege , George Robertson had appointed Shuja-ul-Mulk , who was a', 'bright', 'boy only 12 years old and the youngest surviving son of Aman-ul-Mulk , as the ruler of Chitral .'], 1), ['intelligent', 'bright', 'clever', 'smart'])

rankings = solve(s)
for idx, ranking in enumerate(rankings):
    out = sorted(ranking, key=len)
    out = map(lambda x: '{' + x + '}', out)
    out = ' '.join(out)
    print 'Sentence {} rankings: {}'.format(idx + 1, out)
    
