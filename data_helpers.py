__author__ = 'liming-vie'

import os
import cPickle
import numpy as np
from tensorflow.contrib import learn

def tokenizer(iterator):
    for value in iterator:
        yield value.split()

def load_file(data_dir, fname):
    fname = os.path.join(data_dir, fname)
    print 'Loading file %s'%(fname)
    lines = open(fname).readlines()
    return [line.rstrip() for line in lines]

def process_train_file(data_dir, fname, max_length, min_frequency=10):
    """
    Make vocabulary and transform into id files

    Return:
        vocab_file_name
        vocab_dict: map vocab to id
        vocab_size
    """
    fvocab = '%s.vocab%d'%(fname, max_length)
    foutput = os.path.join(data_dir, fvocab)
    if os.path.exists(foutput):
        print 'Loading vocab from file %s'%foutput
        vocab = load_vocab(data_dir, fvocab)
        return fvocab, vocab, len(vocab)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length,
            tokenizer_fn = tokenizer, min_frequency=min_frequency)
    x_text = load_file(data_dir, fname)
    print 'Vocabulary transforming'
    # will pad 0 for length < max_length
    ids = list(vocab_processor.fit_transform(x_text))
    print "Vocabulary size %d"%len(vocab_processor.vocabulary_)
    fid = os.path.join(data_dir, fname+'.id%d'%max_length)
    print 'Saving %s ids file in %s'%(fname, fid)
    cPickle.dump(ids, open(fid, 'wb'), protocol=2)

    print 'Saving vocab file in %s'%foutput
    size = len(vocab_processor.vocabulary_)
    vocab_str = [vocab_processor.vocabulary_.reverse(i) for i in range(size)]
    with open(foutput, 'w') as fout:
        fout.write('\n'.join(vocab_str))

    vocab = load_vocab(data_dir, fvocab)
    return fvocab, vocab, len(vocab)

def load_data(data_dir, fname, max_length):
    """
    Read id file data

    Return:
        data list: [[length, [token_ids]]]
    """
    fname = os.path.join(data_dir, "%s.id%d"%(fname, max_length))
    print 'Loading data from %s'%fname
    ids = cPickle.load(open(fname, 'rb'))
    data=[]
    for vec in ids:
        length = len(vec)
        if vec[-1] == 0:
            length = list(vec).index(0)
        data.append([length, vec])
    return data

def load_vocab(data_dir, fvocab):
    """
    Load vocab
    """
    fvocab = os.path.join(data_dir, fvocab)
    print 'Loading vocab from %s'%fvocab
    vocab={}
    with open(fvocab) as fin:
        for i, s in enumerate(fin):
            vocab[s.rstrip()] = i
    return vocab

def transform_to_id(vocab, sentence, max_length):
    """
    Transform a sentence into id vector using vocab dict
    Return:
        length, ids
    """
    words = sentence.split()
    ret = [vocab.get(word, 0) for word in words]
    l = len(ret)
    l = max_length if l > max_length else l
    if l < max_length:
        ret.extend([0 for _ in range(max_length - l)])
    return l, ret[:max_length]

def make_embedding_matrix(data_dir, fname, word2vec, vec_dim, fvocab):
    foutput = os.path.join(data_dir, '%s.embed'%fname)
    if os.path.exists(foutput):
        print 'Loading embedding matrix from %s'%foutput
        return cPickle.load(open(foutput, 'rb'))

    vocab_str = load_file(data_dir, fvocab)
    print 'Saving embedding matrix in %s'%foutput
    matrix=[]
    for vocab in vocab_str:
        vec = word2vec[vocab] if vocab in word2vec \
                else [0.0 for _ in range(vec_dim)]
        matrix.append(vec)
    cPickle.dump(matrix, open(foutput, 'wb'), protocol=2)
    return matrix

def load_word2vec(data_dir, fword2vec):
    """
    Return:
        word2vec dict
        vector dimension
        dict size
    """
    fword2vec = os.path.join(data_dir, fword2vec)
    print 'Loading word2vec dict from %s'%fword2vec
    vecs = {}
    vec_dim=0
    with open(fword2vec) as fin:
        size, vec_dim = map(int, fin.readline().split())
        for line in fin:
            ps = line.rstrip().split()
            vecs[ps[0]] = map(float, ps[1:])
    return vecs, vec_dim, size

if __name__ == '__main__':
    data_dir = './data/'
    query_max_length, reply_max_length = [20, 30]
    fquery, freply = []
    fqword2vec, frword2vec = []

    process_train_file(data_dir, fquery, query_max_length)
    process_train_file(data_dir, freply, reply_max_length)

    fqvocab = '%s.vocab%d'%(fquery, query_max_length)
    frvocab = '%s.vocab%d'%(freply, reply_max_length)

    word2vec, vec_dim, _ = load_word2vec(data_dir, fqword2vec)
    make_embedding_matrix(data_dir, fquery, word2vec, vec_dim, fqvocab)

    word2vec, vec_dim, _ = load_word2vec(data_dir, frword2vec)
    make_embedding_matrix(data_dir, freply, word2vec, vec_dim, frvocab)
    pass
