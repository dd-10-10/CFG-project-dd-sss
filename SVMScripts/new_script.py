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
    atac= []
    ids= []
    vecs= []
    for record in SeqIO.parse(file, "fasta"):
        kmer= seq_kmer_count(k, str(record.seq))
        vecs.append(kmer)
        ids.append(record.id[-1])
        atac.append(record.id[-4])
    vecs= np.array(vecs)
    return atac, ids, vecs

def kmer_svm(file, k):
    '''
    Function to perform k-mer svm with data in file
    '''
    atac, ids, vecs= seq_vec(file, k)
    mmc= svm.SVC(kernel= "linear")
    mmc.fit(vecs, ids)
    return mmc

def test(file, t_file, k):
    '''
    Test function
    '''
    clfr= kmer_svm(file, k)
    t_atac, t_ids, t_vecs= seq_vec(t_file, k)
    t_pred= clfr.predict(t_vecs)
    match= t_pred== t_ids
    acc= match.sum()/len(match)
    return acc, t_ids

if __name__== "__main__":
    file= "data/custom_test_seqs.fa"
    t_file= "data/custom_test_seqs.fa"
    k= 2
    print(test(file, t_file, k)[0])