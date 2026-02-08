# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 20:31:18 2026

@authors: Dheeraj, Suchet
"""

#Libraries!
import numpy as np
import pandas as pd
from pathlib import Path

def seq_to_pos(s):
    '''
    Takes a sequence as input, interprets it as a base-4 number, and returns the number in base-10.
    
    Argument:
        :param s: Input sequence
    
    Output:
        Representation of s in base-10
    '''
    base_dict= {"A": 0, "C": 1, "G": 2, "T": 3} #Encoding nucleotides as base-4 digits
    pos= 0
    for i,v in enumerate(s[::-1]):
        pos+= base_dict[v]*(4**i)
    return pos

def markov(in_fasta_path: str, out_dir_path: str, m: int, tf: str):
    '''
    Returns an unnormalised MLE matrix of a markov model of the specified order.
    
    Argument:
        :param in_fasta_path: Fasta file path
        :param out_dir_path: path to Directory to save matrix files 
        :param m: Order of the Markov model
        :param tf: Transciption factor to consider
    
    Outputs:
        Saves Count matrices for the bound and unbound states under a Markov model of the specified order
    '''
    base_dict= {"A": 0, "C": 1, "G": 2, "T": 3}
    tf_dict= {"ATAC": 0, "CTCF": 1, "REST":2, "EP300":3}

    u_arr= np.zeros((4**m, 4), dtype=np.uint32)
    b_arr= np.zeros((4**m, 4), dtype=np.uint32)

    with open(in_fasta_path, "r") as f:
        for line in f:
            if line[0]== ">":
                mode= line.split('_')[-1][tf_dict[tf]] #Example name format: '>chr1:1000-1200_UUBB'
            else:
                for i in range(len(line)- (m+1)):
                    hist= line[i:i+m]
                    row= seq_to_pos(hist) #Row index of each k-mer in the arrays is the seq_to_pos output for that k-mer
                    col= base_dict[line[i+m]] #Columns are in the order A, C, G, T
                    if mode== "U":
                        u_arr[row, col]+= 1
                    else:
                        b_arr[row, col]+= 1
    np.save(out_dir_path / f"{in_fasta_path.stem}_{tf}_u_{m}.npy", u_arr)
    np.save(out_dir_path / f"{in_fasta_path.stem}_{tf}_b_{m}.npy", b_arr)
    return

def main():
    m = 10
    markov(m, "data/fasta/chr1_200bp_bins.fasta", "CTCF")
    
    print("Markov MLE arrays saved.")
    return

if __name__ == "__main__":
    main()
