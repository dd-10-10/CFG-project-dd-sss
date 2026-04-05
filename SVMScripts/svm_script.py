import numpy as np
import pandas as pd
import sklearn.svm as svm
from Bio import SeqIO
from globals import seq_to_pos


def seq_kmer_count(k, seq):
    '''
    Function to count k-mers in seq
    '''
    l = len(seq)
    counts = np.zeros(4**k)
    for i in range(l-k):
        counts[seq_to_pos(seq[i:i+k])] += 1
    return counts

def seq_vec(file, k):
    '''
    Function to make vectors from k-mers for sequences in file
    '''
    ids= []
    vecs= []
    for record in SeqIO.parse(file, "fasta"):
        kmer= seq_kmer_count(k, str(record.seq))
        vec= np.zeros(4**k + 1)
        vec[:-1]= kmer
        if record.id[-4] == "B":
            vec[-1]= 1
        else:
            vec[-1]= 0
        vecs.append(vec)
        ids.append(record.id[-1])
    ids = np.array(ids)
    vecs= np.array(vecs, dtype= np.float16)
    return ids, vecs

def kernel(v1, v2, w):
    '''
    Custom kernel function, to calculate similarity between v1 and v2 as dot product for all non-ATAC dimensions,
    then use XOR to calculate ATAC similarity and weight it by w.
    '''
    x= np.dot(v1[:, :-1], v2[:, :-1].T)
    x/= v1.shape[1]-1
    y= (1 - abs(v1[:, -1] - v2[:, -1])) * w
    return x+y

def kmer_svm(file, k, w):
    '''
    Function to perform k-mer svm with data in file
    '''
    ids, vecs= seq_vec(file, k)
    mmc= svm.SVC(kernel= lambda v1, v2: kernel(v1, v2, w))
    mmc.fit(vecs, ids)
    return mmc

def test(file, t_file, k, w):
    '''
    Test function
    '''
    clfr= kmer_svm(file, k, w)
    t_ids, t_vecs= seq_vec(t_file, k)
    t_pred= clfr.predict(t_vecs)
    match= t_pred== t_ids
    acc= match.sum()/len(match)
    return acc, t_ids

if __name__== "__main__":
    file= "data/temp/chr4_200bp_bins_CTCF_5.fa"
    t_file= "data/temp/chr4_200bp_bins_CTCF_5.fa"
    k= 2
    w= 10
    print(test(file, t_file, k, w)[0])