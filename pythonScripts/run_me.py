import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pathlib import Path
from create_cv_folds import cross_validate
from markov_mle import markov
from markov_score import score
from tsv_to_fasta import tsv_to_fasta

def markov_predict(in_tsv_path: str, out_dir_path: str, m: int, k: int, tf: str):
    genome= "hg38.fa"
    names = cross_validate(in_tsv_path, out_dir_path, k, tf, random_state=42)

    b_new= np.zeros((4**m, 4))
    u_new= np.zeros((4**m, 4))
    for name in names:
        tsv_to_fasta(out_dir_path / name, genome)
        markov(out_dir_path / name, out_dir_path, m, tf)
        b_new+= np.load(out_dir_path / f"{name}_{tf}_b_{m}.npy")
        u_new+= np.load(out_dir_path / f"{name}_{tf}_u_{m}.npy")
    
    roc_auc_list= []
    prc_auc_list= []
    for unname in names: # fold1
        b_arr= b_new- np.load(out_dir_path / f"{unname}_{tf}_b_{m}.npy")
        u_arr= u_new- np.load(out_dir_path / f"{unname}_{tf}_u_{m}.npy")

        scores, true= score(out_dir_path / name, b_arr, u_arr, m, tf)
        fpr, tpr, thresh1= roc_curve(true, scores, pos_label="B")
        prec, rec, thresh2= precision_recall_curve(true, scores, pos_label="B")
        roc_auc= auc(fpr, tpr)
        roc_auc_list.append(roc_auc)
        prc_auc= auc(rec, prec)
        prc_auc_list.append(prc_auc)

        plt.plot(fpr, tpr)
        plt.show()
        plt.plot(rec, prec)
        plt.show()
    
    roc_avg= sum(roc_auc_list)/k
    prc_avg= sum(prc_auc_list)/k
    print("ROC average area= ", roc_avg)
    print("PRC average area= ", prc_avg)

    