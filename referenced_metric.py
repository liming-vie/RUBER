__author__ = 'liming-vie'

import math
import numpy as np
import data_helpers

class Referenced():
    """Referenced Metric
    Measure the similarity between the groundtruth reply and generated reply
    use cosine score.
    Provide three pooling methods for generating sentence vector:
        [max_min | avg | all]
    """
    def __init__(self, data_dir, fword2vec, pooling_type='max_min'):
        """
        Args:
            data_dir:
            fword2vec: word2vec text file
            pooling_type: [max_min | avg | all], default max_min
        """
        self.word2vec, self.vec_dim, _ = data_helpers.load_word2vec(
                data_dir, fword2vec)
        if pooling_type=='max_min':
            self.pooling = self.max_min_pooling
        elif pooling_type=='avg':
            self.pooling = self.average_pooling
        else:
            self.pooling = self.all_pooling

    def __zeroes_vector(self):
        return [1e-10 for _ in xrange(self.vec_dim)]

    def __vector(self, word):
        return self.word2vec[word] if word in self.word2vec \
                else self.__zeroes_vector()

    def sentence_vector(self, sentence):
        sentence = sentence.rstrip().split()
        ret = [self.__vector(word) for word in sentence]
        if len(ret) == 0:
            return [self.__zeroes_vector()]
        return ret

    def max_min_pooling(self, sentence):
        svector = self.sentence_vector(sentence)
        maxp = [max([vec[i] for vec in svector]) for i in range(self.vec_dim)]
        minp = [min([vec[i] for vec in svector]) for i in range(self.vec_dim)]
        return np.concatenate((maxp, minp), axis=0)

    def average_pooling(self, sentence):
        svector = self.sentence_vector(sentence)
        l=float(len(svector))
        return [sum([vec[i] for vec in svector])/l for i in range(self.vec_dim)]

    def all_pooling(self, sentence):
        return np.concatenate((self.max_min_pooling(sentence),
                self.average_pooling(sentence)), axis=0)

    def score(self, groundtruth, generated):
        v1=list(self.pooling(groundtruth))
        v2=list(self.pooling(generated))
        a=sum(v1[i]*v2[i] for i in range(len(v1)))
        b=math.sqrt(sum(i**2 for i in v1)) * math.sqrt(sum(i**2 for i in v2))
        return a/b

    def scores(self, data_dir, fgroundtruth, fgenerated):
        groundtruth = data_helpers.load_file(data_dir, fgroundtruth)
        generated = data_helpers.load_file(data_dir, fgenerated)
        ret = []
        for t, g in zip(groundtruth, generated):
            ret.append(self.score(t, g))
        return ret
