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

def markov(m, path):
    global base_dict
    u_arr= np.zeros((4**m, 4))
    b_arr= np.zeros((4**m, 4))
    with open(path, "r") as f:
        for line in f:
            if line[0]== ">":
                mode= line[-1]
            else:
                for i in range(len(line)- m):
                    hist= line[i:i+m]
                    row= seq_to_pos(hist)
                    col= base_dict[line[i+m]]
                    if mode== "U":
                        u_arr[row, col]+= 1
                    else:
                        b_arr[row, col]+= 1
                #for base in line[:m]:

                                          
                
    