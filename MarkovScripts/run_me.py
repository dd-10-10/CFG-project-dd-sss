import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import logging
import tracemalloc

from sklearn.metrics import roc_curve, precision_recall_curve, auc

from globals import genome_path
from create_cv_folds import cross_validate
from markov_mle import markov
from markov_score import score
from tsv_to_fasta import tsv_to_fasta

def markov_predict(in_tsv_path: Path, output_dir: Path, pseudocounts:float, m: int, k: int, tf: str,
                   force_recalculate: bool = False, store_files: bool = True) -> None:
    '''
    Takes an input tsv file of sequence coordinates, use a markov model,
    perform cross validation and plot ROC and Precision-Recall curves,
    and return average auc across all folds
    
    Arguments:
        in_tsv_path: Path to input tsv
        output_dir: Path to a directory to store output files
        pseudocount: value of pseudocount for calculating MAP estimate
        m: Order of the markov model
        k: Number of folds for Cross Validation
        tf: Transcription Factor to consider
        force_recalculate: Whether to recreate fasta files and recompute count matrices (Default: False)
        store_files: Whether to keep to delete intermediate fasta and matrix files created (Default: False)
    
    Returns:
        None
    '''
    temp_dir_path = (output_dir / f"TempFiles_{in_tsv_path.stem.split('_')[0]}_{m}_{k}_{tf}")
    output_dir = output_dir / "Output"
    temp_dir_path.mkdir(parents=True, exist_ok=True) # Create the folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True) # Create the folder if it does not exist
    logging.info(f"pc = {pseudocounts}, m = {m}, k = {k}, tf = {tf}")
    logging.info("\tCreating Folds")
    names = cross_validate(in_tsv_path, temp_dir_path, k, tf) # Names of fold fasta files created

    logging.info("\tReading MLE matrices")
    b_new= [np.zeros((4**i, 4)) for i in range(m+1)]
    u_new= [np.zeros((4**i, 4)) for i in range(m+1)]
    for index, name in enumerate(names):
        logging.info(f"\t\tConverting file {index+1} to fasta")
        tsv_to_fasta(temp_dir_path / f"{name}.tsv") # Extract sequence using genome file
        logging.info(f"\t\tCalculating MLE for file {index+1}")
        markov(temp_dir_path / f"{name}.fa", temp_dir_path, m, tf) # Calculate MLE (counts) for each fasta file
        b_new[m]+= np.load(temp_dir_path / f"{name}_b_{m}.npy")
        u_new[m]+= np.load(temp_dir_path / f"{name}_u_{m}.npy")
        for i in range(m):
            markov(temp_dir_path / f"{name}.fa", temp_dir_path, i, tf)
            b_new[i]+= np.load(temp_dir_path / f"{name}_b_{i}.npy")
            u_new[i]+= np.load(temp_dir_path / f"{name}_u_{i}.npy")

    roc_auc_list= []
    prc_auc_list= []
    logging.info("\tStarting Calculation and Inference")
    for fold_index, unname in enumerate(names):
        logging.info(f"\t\tStarting fold {fold_index+1}")
        b_arr= [np.zeros((4**i, 4)) for i in range(m+1)]
        u_arr= [np.zeros((4**i, 4)) for i in range(m+1)]
        for i in range(m+1):
            b_arr[i]= b_new[i]- np.load(temp_dir_path / f"{unname}_b_{i}.npy") # Get MLE matrix for (k-1) folds
            u_arr[i]= u_new[i]- np.load(temp_dir_path / f"{unname}_u_{i}.npy")
            b_arr[i] += pseudocounts
            u_arr[i] += pseudocounts
            b_arr[i] /= b_arr[i].sum(axis=1, keepdims=True)
            u_arr[i] /= u_arr[i].sum(axis=1, keepdims=True)
        
        logging.info(f"\t\t\tCalculating Scores")
        json_path = output_dir / f"Scores_{in_tsv_path.stem}_m={m}_fold={fold_index+1}of{k}_tf={tf}.json"
        if (json_path).exists() and not force_recalculate:
            with open(json_path, 'r') as f:
                data = json.load(f)
                scores = data['scores']; true = data['true']
                fpr = data['fpr']; tpr = data['tpr']
                prec = data['prec']; rec = data['rec']
                prop = true.count('B')/len(true)
        else:
            scores, true = score(temp_dir_path / f"{unname}.fa", b_arr, u_arr, m, tf) # Get log-odds scores for each fold
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
        plt.savefig(output_dir / f"ROC_{in_tsv_path.stem}_m={m}_fold={fold_index+1}of{k}_tf={tf}.png")
        plt.close()
        
        plt.figure(figsize=(8,8), dpi=200)
        plt.plot(rec, prec, 'k-')
        plt.plot([0,1], [prop, prop], 'r--')
        plt.title(f"Precision-Recall Curve\ntf={tf}, m={m}, fold={fold_index+1}/{k}")
        plt.xlabel(r"Recall $(\frac{TP}{TP+FN})$")
        plt.ylabel(r"Precision $(\frac{TP}{TP+FP})$")
        plt.savefig(output_dir / f"PRC_{in_tsv_path.stem}_m={m}_fold={fold_index+1}of{k}_tf={tf}.png")
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
        shutil.rmtree(temp_dir_path)
    logging.info("\tDone!")
    return

def main():
    parser= argparse.ArgumentParser(prog= "pythonScripts/run_me.py",
                description= "A program to predict whether a TF will bind to a sequence")
    parser.add_argument("input", help= "path to input tsv file")
    parser.add_argument("out", help= "path to save output files to")
    parser.add_argument("m", type= int, help= "order of Markov model to train")
    parser.add_argument("k", type= int, help= "number of folds for cross-validation")
    parser.add_argument("tf", choices= ["CTCF", "EP300", "REST"], help= "transcription factor")
    parser.add_argument("-pc", type= float, default= 1, help= "specify pseudocounts (default= 1)")
    parser.add_argument("--force_recalculate", action= "store_false", help= "force recalculation and overwriting of existing MLE matrices and fasta files")
    parser.add_argument("--delete_temp", action= "store_false", help= "delete the temp folder once the program execution is complete")
    args= parser.parse_args()
    in_tsv_path = Path(args.input)
    out_path = Path(args.out)
    m= args.m
    k= args.k
    tf= args.tf
    pc= args.pc
    fr= args.force_recalculate
    s= args.delete_temp

    out_path.mkdir(parents=True, exist_ok=True) # Create the folder if it does not exist
    logging.basicConfig(
        level=logging.INFO,
        filename = out_path / 'logs.txt',
        format='%(asctime)s - %(message)s',
        datefmt='%d-%m-%y %H:%M:%S',
        filemode = 'a'
    )
    
    tracemalloc.start()
    markov_predict(in_tsv_path, out_path, pseudocounts= pc, m= m, k= k, tf= tf, force_recalculate= fr, store_files= s)
    trm = tracemalloc.get_tracemalloc_memory()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logging.info(f"\tPeak Memory Usage = {peak-trm} bytes")

if __name__ == '__main__':
    main()