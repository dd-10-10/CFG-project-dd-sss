import numpy as np
import pandas as pd
from pathlib import Path
from globals import base_dict, tf_dict, seq_to_pos

def score(in_fasta_path: str, b_arr: list[np.ndarray], u_arr: list[np.ndarray], m: int, tf: str) -> tuple[list[float], list[str]]:
    '''
    Calculate log-odds score for sequences in a fasta file using transition probability matrices
    
    Arguments:
        in_fasta_path: Path to the input fasta files
        b_arr: Transition probability matrix for bound sequences
        u_arr: Transition probability matrix for unbound sequences
        m: Order of Markov model
        tf: Transcription Factor to consider
    
    Returns:
        Tuple of (List of predicted log-odds scores, List of true labels)
    '''

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
                    u_score+= np.log10(u_arr[-1][row, col])
                    b_score+= np.log10(b_arr[-1][row, col])
                    if i< m:
                        hist= line[:i+1]
                        row= seq_to_pos(hist)
                        col= base_dict[line[i+1]]
                        u_score+= np.log10(u_arr[i][row, col])
                        b_score+= np.log10(b_arr[i][row, col])
                true_l.append(mode)
                scores_l.append(b_score- u_score)
    return scores_l, true_l