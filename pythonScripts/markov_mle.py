import numpy as np
import pandas as pd
from pathlib import Path
from globals import seq_to_pos, base_dict, tf_dict

def markov(in_fasta_path: Path, out_dir_path: Path, m: int, tf: str) -> None:
    '''
    Create and save an unnormalised count matrix for a markov model of the specified order.
    
    Arguments:
        in_fasta_path: Fasta file path
        out_dir_path: Path to Directory to save matrix files 
        m: Order of the Markov model
        tf: Transciption Factor
    
    Outputs:
        None
    '''
    if (out_dir_path / f"{in_fasta_path.stem}_u_{m}.npy").exists() and (out_dir_path / f"{in_fasta_path.stem}_b_{m}.npy").exists():
        print(f"{in_fasta_path.stem}_u_{m}.npy, {in_fasta_path.stem}_b_{m}.npy exists, skipping")
        return
    b_arr= np.zeros((4**m, 4))
    u_arr= np.zeros((4**m, 4))

    with open(in_fasta_path, "r") as f:
        for line in f:
            if line[0]== ">":
                mode= line.split('_')[-1][tf_dict[tf]] # Example name format: '>chr1:1000-1200_UUBB'
            else:
                for i in range(len(line)- (m+1)):
                    hist= line[i:i+m]
                    row= seq_to_pos(hist) # Row index of each k-mer in the arrays is the seq_to_pos output for that k-mer
                    col= base_dict[line[i+m]] # Columns are in the order A, C, G, T
                    if mode == "U":
                        u_arr[row, col]+= 1
                    else:
                        b_arr[row, col]+= 1
    np.save(out_dir_path / f"{in_fasta_path.stem}_u_{m}.npy", u_arr)
    np.save(out_dir_path / f"{in_fasta_path.stem}_b_{m}.npy", b_arr)
    return

def main():
    m = 10
    markov(m, "data/fasta/chr1_200bp_bins.fasta", "CTCF")
    
    print("Markov MLE arrays saved.")
    return

if __name__ == "__main__":
    main()
