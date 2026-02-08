import numpy as np
import pandas as pd
from pathlib import Path
from markov_mle import seq_to_pos

def score(in_fasta_path: str, b_arr: np.ndarray, u_arr: np.ndarray, m: int, tf: str):
    '''
    Returns:
        List of log-odds scores for each sequence,
        List of true labels for each sequence
    '''
    base_dict= {"A": 0, "C": 1, "G": 2, "T": 3}
    tf_dict= {"ATAC": 0, "CTCF": 1, "REST":2, "EP300":3}

    scores_l= []
    true_l= []
    with open(in_fasta_path, "r") as f:
        for line in f:
            if line[0]== ">":
                mode= line.split('_')[-1][tf_dict[tf]] #Example name format: '>chr1:1000-1200_UUBB'
            else:
                u_score= 0
                b_score= 0
                for i in range(len(line)- (m+1)):
                    hist= line[i:i+m]
                    row= seq_to_pos(hist) #Row index of each k-mer in the arrays is the seq_to_pos output for that k-mer
                    col= base_dict[line[i+m]] #Columns are in the order A, C, G, T
                    true_l.append(mode)
                    u_score+= np.log10(u_arr[row, col])
                    b_score+= np.log10(b_arr[row, col])
                    scores_l.append(b_score- u_score)
    return scores_l, true_l