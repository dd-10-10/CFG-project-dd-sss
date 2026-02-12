import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json

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
    
    Returns:
        None
    '''
    output_dir = (out_dir_path.parent / "Output")
    out_dir_path.mkdir(parents=True, exist_ok=True) # Create the folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True) # Create the folder if it does not exist
    logging.basicConfig(
        level=logging.INFO,
        filename = output_dir / 'logs.txt',
        format='%(asctime)s - %(message)s',
        datefmt='%d-%m-%y %H:%M:%S',
        filemode = 'w'
    )
    logging.info(f"pc = {pseudocounts}, m = {m}, k = {k}, tf = {tf}")
    logging.info("\tCreating Folds")
    names = cross_validate(in_tsv_path, out_dir_path, k, tf, random_state=42) # Names of fold fasta files created

    logging.info("\tReading MLE matrices")
    b_new= np.zeros((4**m, 4))
    u_new= np.zeros((4**m, 4))
    for index, name in enumerate(names):
        logging.info(f"\t\tConverting file {index+1} to fasta")
        tsv_to_fasta(out_dir_path / f"{name}.tsv") # Extract sequence using genome file
        logging.info(f"\t\tCalculating MLE for file {index+1}")
        markov(out_dir_path / f"{name}.fa", out_dir_path, m, tf) # Calculate MLE (counts) for each fasta file
        b_new+= np.load(out_dir_path / f"{name}_b_{m}.npy")
        u_new+= np.load(out_dir_path / f"{name}_u_{m}.npy")
    
    roc_auc_list= []
    prc_auc_list= []
    logging.info("\tStarting Calculation and Inference")
    for fold_index, unname in enumerate(names):
        logging.info(f"\t\tStarting fold {fold_index+1}")
        b_arr= b_new- np.load(out_dir_path / f"{unname}_b_{m}.npy") # Get MLE matrix for (k-1) folds
        u_arr= u_new- np.load(out_dir_path / f"{unname}_u_{m}.npy")
        b_arr += pseudocounts
        u_arr += pseudocounts
        b_arr /= b_arr.sum(axis=1, keepdims=True)
        u_arr /= u_arr.sum(axis=1, keepdims=True)
        
        logging.info(f"\t\t\tCalculating Scores")
        json_path = output_dir / f"Scores_{in_tsv_path.stem}_m={m}_k={k}_tf={tf}.json"
        if (json_path).exists() and not force_recalculate:
            with open(json_path, 'r') as f:
                data = json.load(f)
                scores = data['scores']; true = data['true']
                fpr = data['fpr']; tpr = data['tpr']
                prec = data['prec']; rec = data['rec']
        else:
            scores, true = score(out_dir_path / f"{unname}.fa", b_arr, u_arr, m, tf) # Get log-odds scores for each fold
            prop = true.count('B')/len(true)
            fpr, tpr, thresh1= roc_curve(true, scores, pos_label="B")
            prec, rec, thresh2= precision_recall_curve(true, scores, pos_label="B")
            with open(json_path, 'w') as f:
                json.dump({'scores':list(scores), 'true':list(true), 'fpr':list(fpr), 'tpr':list(tpr),
                        'prec':list(prec), 'rec':list(rec)}, fp=f)
        roc_auc= auc(fpr, tpr)
        roc_auc_list.append(roc_auc)
        prc_auc= auc(rec, prec)
        prc_auc_list.append(prc_auc)
        
        logging.info(f"\t\t\tPlotting Evaluation Metrics")
        plt.figure(figsize=(8,8), dpi=200)
        plt.plot(fpr, tpr, 'k-')
        plt.plot([0,1],[0,1], 'r--')
        plt.title(f"ROC Curve\ntf={tf}, m={m}, fold={fold_index+1}/{k}")
        plt.xlabel(r"False Positive Rate $(\frac{FP}{FP+TN})$")
        plt.xlabel(r"True Positive Rate $(\frac{TP}{TP+FN})$")
        plt.savefig(output_dir / f"ROC_{in_tsv_path.stem}_m={m}_fold={fold_index+1}-{k}_tf={tf}.png")
        plt.close()
        
        plt.figure(figsize=(8,8), dpi=200)
        plt.plot(rec, prec, 'k-')
        plt.plot([0,1], [prop, prop], 'r--')
        plt.title(f"Precision-Recall Curve\ntf={tf}, m={m}, fold={fold_index+1}/{k}")
        plt.xlabel(r"Recall $(\frac{TP}{TP+FN})$")
        plt.ylabel(r"Precision $(\frac{TP}{TP+FP})$")
        plt.savefig(output_dir / f"PRC_{in_tsv_path.stem}_m={m}_fold={fold_index+1}-{k}_tf={tf}.png")
        plt.close()
    
    logging.info("\tAveraging across folds")
    roc_avg= sum(roc_auc_list)/k
    prc_avg= sum(prc_auc_list)/k
    with open(output_dir / f"Area_{in_tsv_path.stem}_m={m}_k={k}_tf={tf}.txt", 'w') as f:
        print("ROC average area= ", roc_avg, file=f)
        print("PRC average area= ", prc_avg, file=f)
        print("ROC areas= ", *roc_auc_list, sep='\t', file=f)
        print("PRC areas= ", *prc_auc_list, sep='\t', file=f)
    
    if not store_files:
        logging.info("\tDELETING TEMP FILES")
        import shutil
        shutil.rmtree(out_dir_path)
    logging.info("\tDone!")
    return

def main():
    in_tsv_path = Path('data/tsv/chr1_200bp_bins.tsv')
    out_path = Path('data/temp')
    for m in range(0,1):
        markov_predict(in_tsv_path,out_path, pseudocounts=1, m=m, k=3, tf='CTCF', force_recalculate=True)

if __name__ == '__main__':
    main()