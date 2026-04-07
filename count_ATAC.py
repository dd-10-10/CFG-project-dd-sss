import os
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from Bio import SeqIO

def count(FOLDER):
    dir = {"".join(i):0 for i in product(['U', 'B'], ['U', 'B'], ['U', 'B'], ['U', 'B'])}
    for file in [file for file in os.listdir(FOLDER) if file.endswith('bins.tsv')]:
        df = pd.read_csv(FOLDER / file, sep='\t')
        t = df[['ATAC', 'CTCF', 'REST', 'EP300']].values
        for ind in t:
            dir["".join(ind)]+=1
        #print(file, '\n', dir.items())

    print(dir.items())

def separate(file, tf, atac):
    file_ctcf  = open("data/tsv/CTCF_bound.fa" , 'w')
    file_rest  = open("data/tsv/REST_bound.fa" , 'w')
    file_ep300 = open("data/tsv/EP300_bound.fa", 'w')
    try:
        for index, record in enumerate(SeqIO.parse(file, 'fasta')):
            if index % 100_000 == 0:
                print(datetime.now(), index)
            if record.id[-1] == 'B':
                file_ep300.write(f">{record.id}\n{record.seq}\n")
            if record.id[-2] == 'B':
                file_rest.write(f">{record.id}\n{record.seq}\n")
            if record.id[-3] == 'B':
                file_ctcf.write(f">{record.id}\n{record.seq}\n")
    finally:
        print('Closing Files')
        file_ctcf.close()
        file_rest.close()
        file_ep300.close()
        
def merge_preds():
    chr = [3, 10, 17]
    with open ('preds_ep300.txt', 'r') as preds_file:
        preds = "".join([i.strip() for i in preds_file.read()])
    print(len(preds))
    
    pred_index = 0
    for i in chr:
        with open(f"data/tsv/chr{i}_200bp_bins_unknown.fa") as f:
            seqs = np.array(f.readlines()[1::2])
        mask = np.array(['n' not in i.lower() for i in seqs])
        print(mask.sum(), end=' ')

        file = Path(f"chr{i}_200bp_bins_unknown_predictions.tsv")
        df = pd.read_csv(file, sep='\t')
        print(df.shape, end=' ')
        df ['EP300'] = np.zeros_like(df['ATAC'])
        for index in df.index:
            if mask[index]:
                df.at[index, 'EP300'] = {'U':0, 'B':1}[preds[pred_index]]
                pred_index += 1
            else:
                print(".", end='')
                df.at[index, 'EP300'] = None
        df.to_csv(Path(f"chr{i}_200bp_bins_unknown_predictions_full.tsv"), index=False, sep='\t')
        print(pred_index)

if __name__ == '__main__':
    #file = Path('data/tsv/chrAll.fa')
    #t = count(file.parent)
    merge_preds()
