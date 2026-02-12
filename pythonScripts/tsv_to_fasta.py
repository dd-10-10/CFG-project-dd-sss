from pathlib import Path
from datetime import datetime
import pybedtools
import os
import pandas as pd
from globals import genome_path

def tsv_to_fasta(in_tsv_path:Path, force_recalculate: bool = False) -> None:
    '''
    Convert a TSV file with genomic coordinates to a FASTA file using a reference genome.

    Arguments:
        in_tsv: Path to the input TSV file
        force_recalculate: Whether to recreate fasta files (Default: False)

    Returns:
        None
    '''
    out_fa = in_tsv_path.parent / f"{in_tsv_path.stem}.fa"
    if out_fa.exists() and not force_recalculate:
        return
    df = pd.read_csv(in_tsv_path, sep="\t", dtype={'chr':str, 'start':int, 'end':int, "ATAC":str, 'CTCF':str, "REST":str, 'EP300':str})
    df['name'] = df['chr'] + ":" + df['start'].astype(str) + "-" + df['end'].astype(str) + "_" + df['ATAC'] + df['CTCF'] + df['REST'] + df['EP300']
    df = df[['chr', 'start', 'end', 'name']]
    a = pybedtools.BedTool(df.values.tolist())
    a = a.sequence(fi=genome_path, fo = out_fa, nameOnly=True)
    return

def main():
    DATA_FOLDER = Path("data/tsv")
    GENOME_FASTA = Path("data/hg38.fa")
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".tsv"):
            tsv_to_fasta(TSV_FOLDER / file)
    return

if __name__ == "__main__":
    main()
