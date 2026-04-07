from pathlib import Path
import numpy as np
from Bio import SeqIO, motifs
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import time
import pandas as pd

base_dict = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}

import numpy as np

# Create a global lookup table (256 ASCII characters)
# Unrecognized characters will just return [0,0,0,0]
_LOOKUP_TABLE = np.zeros((256, 4), dtype=np.float16)
_LOOKUP_TABLE[ord('A')] = _LOOKUP_TABLE[ord('a')] = [1.0, 0.0, 0.0, 0.0]
_LOOKUP_TABLE[ord('C')] = _LOOKUP_TABLE[ord('c')] = [0.0, 1.0, 0.0, 0.0]
_LOOKUP_TABLE[ord('G')] = _LOOKUP_TABLE[ord('g')] = [0.0, 0.0, 1.0, 0.0]
_LOOKUP_TABLE[ord('T')] = _LOOKUP_TABLE[ord('t')] = [0.0, 0.0, 0.0, 1.0]

def one_hot(seq: str) -> np.ndarray:
    # Convert the string to a numpy array of ASCII integer values
    # np.frombuffer is practically instantaneous
    seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    
    # Use the ASCII values as indices to fetch from the lookup table
    return _LOOKUP_TABLE[seq_bytes]

class model(nn.Module):
    def __init__(self, pwm):
        super(model, self).__init__()
        self.pwm = torch.tensor(pwm, dtype=torch.float16)
        self.conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=self.pwm.shape[1], bias=False)
        self.conv.weight = nn.Parameter(self.pwm.unsqueeze(0))
        self.maxpool = nn.AdaptiveMaxPool1d(1) # Keep top score only

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x.view(-1, 1)

def scan_motifs(sequence, pwm):
    # Convert the sequence to one-hot encoding
    one_hot_seq = np.array([one_hot(i) for i in sequence])
    one_hot_seq = torch.tensor(one_hot_seq, dtype=torch.float16).permute(0, 2, 1).to('cuda')
    
    # Create the model and scan for motifs
    motif_model = model(pwm).to('cuda')
    motif_model.eval()
    
    scores = motif_model(one_hot_seq).detach().to('cpu').flatten().numpy()
    return scores

def read_fasta(file_path):
    sequences = []
    ids = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    return sequences, ids

def main():
    start = time.time()
    train = False
    if train:
        file = Path('data/tsv/chr6_200bp_bins.fa')
    else:
        file = Path('data/tsv/chr3_200bp_bins_unknown.fa')
        file = Path('data/tsv/chr10_200bp_bins_unknown.fa')
        file = Path('data/tsv/chr17_200bp_bins_unknown.fa')
    
    print('Reading sequences from file...')
    sequences, ids = read_fasta(file)
    ids = [i[-3] for i in ids]
    print(f'Read {len(sequences)} sequences.')
    
    print('Parsing motif from JASPAR')
    with open("data/JASPAR/MA0138.2.pfm") as handle: # MA0139.2 for CTCF, MA0138.2 for REST
        motif = motifs.read(handle, "pfm")
    
    pwm = motif.counts.normalize(pseudocounts=0.1).log_odds()
    pwm = np.array([pwm[n] for n in "ACGT"])
    # print(pwm)
    
    print('Calculating motif scores')
    scores = scan_motifs(sequences, pwm)
    print(scores.shape, 'scores calculated.')
    
    if train:
        print('Evaluating')
        fpr, tpr, _ = roc_curve(ids, scores, pos_label='B')
        precision, recall, _ = precision_recall_curve(ids, scores, pos_label='B')
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'PR AUC: {pr_auc:.4f}')
    else:
        print('Saving scores to CSV')
        output_file = file.parent / f"{file.stem}_predictions.tsv"
        if not output_file.exists():
            df = pd.read_csv(file.parent / f"{file.stem}.tsv", sep='\t')
            df.to_csv(output_file, index=False, sep='\t')
        df = pd.read_csv(file.parent / f"{file.stem}_predictions.tsv", sep='\t')
        df['REST'] = scores
        df.to_csv(file.parent / f"{file.stem}_predictions.tsv", index=False, sep='\t')
    print((max(scores), min(scores)))
    end = time.time()
    print(f'Time taken: {end - start:.2f} seconds')
    
if __name__ == "__main__":
    main()
