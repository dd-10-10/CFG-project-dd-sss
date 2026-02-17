from pathlib import Path
import argparse
from globals import seq_to_pos, base_dict

import numpy as np

def simpler_model(in_fasta_path:Path, m: int) -> None:
    '''
    Takes in a fasta file of sequences, trains a markov model of order 'm' on them, then
    calculates log-odds scores of the sequences on an MAP trained using the same sequences.
    
    Arguments:
        in_fasta_path: Path to input fasta file
        m: Order of the markov model
    
    Returns:
        (Prints and Returns) Numpy array of log-odds scores
    '''
    pseudocount = 0.1
    count_arr = np.zeros((4**m, 4))
    seq = ""
    with open(in_fasta_path, "r") as f:
        for line in f:
            if line[0] == ">":
                if seq:
                    for i in range(len(seq)- (m+1)):
                        hist = seq[i:i+m]
                        row = seq_to_pos(hist) # Row index of each k-mer
                        col = base_dict[seq[i+m]] # Columns in the order A, C, G, T
                        count_arr[row, col] += 1
                seq = ""
            else:
                seq += line.strip()
        for i in range(len(seq)- (m+1)):
                    hist = seq[i:i+m]
                    row = seq_to_pos(hist) # Row index of each k-mer
                    col = base_dict[seq[i+m]] # Columns in the order A, C, G, T
                    count_arr[row, col] += 1
    count_arr += pseudocount
    count_arr /= count_arr.sum(axis=1, keepdims=True)
    
    
    scores = []
    seq = ""
    with open(in_fasta_path, "r") as f:
        for line in f:
            if line[0] == ">":
                if seq:
                    score = 0
                    for i in range(len(seq)- (m+1)):
                        hist = seq[i:i+m]
                        row = seq_to_pos(hist) #Row index of each k-mer in the arrays is the seq_to_pos output for that k-mer
                        col = base_dict[seq[i+m]] #Columns are in the order A, C, G, T
                        score += np.log10(count_arr[row, col])
                    scores.append(score)
                seq = ""
            else:
                seq += line.strip()
        score = 0
        for i in range(len(seq)- (m+1)):
            hist = seq[i:i+m]
            row = seq_to_pos(hist) #Row index of each k-mer in the arrays is the seq_to_pos output for that k-mer
            col = base_dict[seq[i+m]] #Columns are in the order A, C, G, T
            score += np.log10(count_arr[row, col])
        scores.append(score)
    scores = np.array(scores)
    print(*scores, sep='\n')
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help = "Path to input fasta file")
    parser.add_argument('m', help = "Order of the markov model", type=int)
    args = parser.parse_args()
    in_path = Path(args.in_path)
    scores = simpler_model(in_path, args.m)
