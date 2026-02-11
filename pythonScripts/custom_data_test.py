from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from create_cv_folds import cross_validate
from globals import base_dict, seq_to_pos
from markov_mle import markov
from markov_score import score
from tsv_to_fasta import tsv_to_fasta

def create_sequence_from_tpm(tpm: np.ndarray[float]) -> str:
    rng = np.random.default_rng()
    bases = np.array(['A', 'C', 'G', 'T'])
    current = rng.integers(4)
    seq = [bases[current]]
    
    for _ in range(199):
        current = rng.choice(4, p=tpm[current])
        seq.append(bases[current])
    return ''.join(seq)

def main():
    fasta_path = Path("data/custom_test_seqs.fa")
    M = 1
    BOUND_SEQS = 200
    UNBOUND_SEQS = 800
    b_arr = np.array([ #ACGT
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])+abs(np.random.normal(loc=0.1, scale=0.005, size=(4,4)))
    u_arr = np.array([ # TGCA
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])+abs(np.random.normal(loc=0.15, scale=0.005, size=(4,4)))
    b_arr /= b_arr.sum(axis=1, keepdims=True)
    u_arr /= u_arr.sum(axis=1, keepdims=True)
    print(u_arr, b_arr, sep='\n')
    with open(fasta_path, 'w') as f:
        f.writelines([f">{i}_BBBB\n"+create_sequence_from_tpm(b_arr)+"\n" for i in range(BOUND_SEQS)])
    with open(fasta_path, 'a') as f:
        f.writelines([f">{i}_UUUU\n"+create_sequence_from_tpm(u_arr)+"\n" for i in range(UNBOUND_SEQS)])
    
    scores, true= score(fasta_path, b_arr, u_arr, M, "CTCF")
    plt.hist(scores[:200], bins = 100, color='b', alpha=0.5)
    plt.hist(scores[200:], bins = 100, color='r', alpha=0.5)
    plt.show()
    
    fpr, tpr, thresh1= roc_curve(true, scores, pos_label="B")
    prec, rec, thresh2= precision_recall_curve(true, scores, pos_label="B")
    print("ROC_AUC =", auc(fpr, tpr))
    print("PRC_AUC =", auc(rec, prec))
    
    plt.plot(fpr, tpr)
    plt.show()
    plt.plot(rec, prec)
    plt.show()
    return

if __name__ == '__main__':
    main()