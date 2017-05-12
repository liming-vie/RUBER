__author__ = 'liming-vie'

import os
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced

class Hybrid():
    def __init__(self,
            data_dir,
            frword2vec,
            fqembed,
            frembed,
            qmax_length=20,
            rmax_length=30,
            ref_method='max_min',
            gru_units=128, mlp_units=[256, 512, 128]
        ):

        self.ref=Referenced(data_dir, frword2vec, ref_method)
        self.unref=Unreferenced(qmax_length, rmax_length,
                os.path.join(data_dir,fqembed),
                os.path.join(data_dir,frembed),
                gru_units, mlp_units,
                train_dir=train_dir)

    def train_unref(self, data_dir, fquery, freply):
        self.unref.train(data_dir, fquery, freply)

    def normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret

    def scores(self, data_dir, fquery ,freply, fgenerated, fqvocab, frvocab):
        ref_scores = self.ref.scores(data_dir, freply, fgenerated)
        ref_scores = self.normalize(ref_scores)

        unref_scores = self.unref.scores(data_dir, fquery, fgenerated,
                fqvocab, frvocab)
        unref_socres = self.normalize(unref_scores)

        return [min(a,b) for a,b in zip(ref_scores, unref_scores)]

if __name__ == '__main__':
    train_dir = ''
    data_dir = 'data/'
    qmax_length, rmax_length = [20, 30]
    fquery, freply = []
    frword2vec = ''

    hybrid = Hybrid(data_dir, frword2vec, '%s.embed'%fquery, '%s.embed'%freply)
    """test"""
    out_file='word2vec_out'
#    scores = hybrid.unref.scores(data_dir, '%s.sub'%fquery, '%s.sub'%freply, "%s.vocab%d"%(fquery,qmax_length), "%s.vocab%d"%(freply, rmax_length))
    scores = hybrid.scores(data_dir, '%s.sub'%fquery, '%s.true.sub'%freply, out_file, '%s.vocab%d'%(fquery, qmax_length),'%s.vocab%d'%(freply, rmax_length))
    for i, s in enumerate(scores):
        print i,s
    print 'avg:%f'%(sum(scores)/len(scores))

    """train"""
#    hybrid.train_unref(data_dir, fquery, freply)
