# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 20:31:18 2026

@author: Dheeraj, Suchet
"""

import numpy as np
import pandas as pd
#import scipy.sparse as ss
import matplotlib.pyplot as plt
import seaborn as sns

base_dict= {"A": 0, "C": 1, "G": 2, "T": 3}

def seq_to_pos(s):
    global base_dict
    pos= 0
    for i,v in enumerate(s[::-1]):
        pos+= base_dict[v]*(4**i)
    return pos

def markov(m, path, tf):
    global base_dict
    tf_dict= {"ATAC": 0, "CTCF": 1, "REST":2, "EP300":3}

    u_arr= np.zeros((4**m, 4), dtype=np.uint32)
    b_arr= np.zeros((4**m, 4), dtype=np.uint32)

    with open(path, "r") as f:
        for line in f:
            if line[0]== ">":
                mode= line.split('_')[-1][tf_dict[tf]]
            else:
                for i in range(len(line)- (m+1)):
                    hist= line[i:i+m]
                    row= seq_to_pos(hist)
                    col= base_dict[line[i+m]]
                    if mode== "U":
                        u_arr[row, col]+= 1
                    else:
                        b_arr[row, col]+= 1
            #TODO: Adaptive markov for first m-1 bases
    return u_arr, b_arr

def main():
    m = 10
    u_arr, b_arr = markov(m, "data/fasta/chr22_200bp_bins.fasta", "CTCF")
    np.save(f"data/markov_mle/ctcf_u_markov_m{m}.npy", u_arr)
    np.save(f"data/markov_mle/ctcf_b_markov_m{m}.npy", b_arr)
    print("Markov MLE arrays saved.")

if __name__ == "__main__":
    main()