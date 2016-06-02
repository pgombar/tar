class Scorer:
    def __init__(self):
        return
    
    def score(self, (sentence, idx), sub):
        assert False
        
    def rank(self, (sentence, idx), subs):
        scores = map(lambda sub: -self.score((sentence, idx), sub), subs)
        subs = sorted(zip(scores, subs))
        return map(lambda (_, sub): sub, subs)

    def rankMultiple(self, tasks):
        return map(lambda ((a, b), c): self.rank((a, b), c), tasks)
