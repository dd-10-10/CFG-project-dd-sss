import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import roc_curve, precision_recall_curve, auc

from globals import genome_path
from create_cv_folds import cross_validate
from markov_mle import markov
from markov_score import score
from tsv_to_fasta import tsv_to_fasta

def markov_predict(in_tsv_path: str, out_dir_path: str, pseudocounts:float, m: int, k: int, tf: str,
                   force_recalculate: bool = False, store_files: bool = True) -> None:
    '''
    Takes an input tsv file of sequence coordinates, use a markov model,
    perform cross validation and plot ROC and Precision-Recall curves,
    and return average auc across all folds
    
    Arguments:
        in_tsv_path: Path to input tsv
        out_dir_path: Path to a directory to store intermediate files
        pseudocount: value of pseudocount for calculating MAP estimate
        m: Order of the markov model
        k: Number of folds for Cross Validation
        tf: Transcription Factor to consider
        force_recalculate: Whether to recreate fasta files and recompute count matrices (Default: False)
        store_files: Whether to keep to delete intermediate fasta and matrix files created (Default: False)
    '''
    out_dir_path.mkdir(parents=True, exist_ok=True) # Create the folder if it does not exist
    names = cross_validate(in_tsv_path, out_dir_path, k, tf, random_state=42) # Names of fold fasta files created

    b_new= np.zeros((4**m, 4))
    u_new= np.zeros((4**m, 4))
    for name in names:
        tsv_to_fasta(out_dir_path / name, genome_path) # Extract sequence using genome file
        markov(out_dir_path / f"{name.stem}.fa", out_dir_path, m, tf) # Calculate MLE (counts) for each fasta file
        b_new+= np.load(out_dir_path / f"{name.stem}_b_{m}.npy")
        u_new+= np.load(out_dir_path / f"{name.stem}_u_{m}.npy")
    
    roc_auc_list= []
    prc_auc_list= []
    for unname in names:
        b_arr= b_new- np.load(out_dir_path / f"{unname.stem}_b_{m}.npy") # Get MLE matrix for (k-1) folds
        u_arr= u_new- np.load(out_dir_path / f"{unname.stem}_u_{m}.npy")
        b_arr += pseudocounts
        u_arr += pseudocounts
        b_arr /= b_arr.sum(axis=1, keepdims=True)
        u_arr /= u_arr.sum(axis=1, keepdims=True)
        
        scores, true= score(out_dir_path / f"{unname.stem}.fa", b_arr, u_arr, m, tf) # Get log-odds scores for each fold
        plt.hist(scores, bins=range(-300, 300, 1))
        plt.show()
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
    print("ROC areas= ", roc_auc_list)
    print("PRC areas= ", prc_auc_list)
    
    if not store_files:
        import shutil
        shutil.rmtree(out_dir_path)

def main():
    in_tsv_path = Path('data/tsv/chr22_200bp_bins.tsv')
    out_path = Path('data/temp')
    markov_predict(in_tsv_path,out_path, pseudocounts=1, m=2, k=3, tf='CTCF')

if __name__ == '__main__':
    main()