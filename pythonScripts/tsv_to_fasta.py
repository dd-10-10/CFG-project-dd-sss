'''
    Convert all TSV files (in BED formatting) to FASTA format using hg38 genome
'''
from pathlib import Path
from datetime import datetime
import pybedtools
import os
import pandas as pd
# Convert TSV to FASTA

def tsv_to_fasta(in_tsv:Path, in_fa:Path, out_fa:Path) -> None:
    '''
    Convert a TSV file with genomic coordinates to a FASTA file using a reference genome.

    Arguments:
        in_tsv: Path to the input TSV file
        in_fa: Path to the genome FASTA file
        out_fa: Path to the output FASTA file
    
    Returns:
        None
    '''
    df = pd.read_csv(in_tsv, sep="\t", dtype={'chr':str, 'start':int, 'end':int, "ATAC":str, 'CTCF':str, "REST":str, 'EP300':str})
    df['name'] = df['chr'] + ":" + df['start'].astype(str) + "-" + df['end'].astype(str) + "_" + df['ATAC'] + df['CTCF'] + df['REST'] + df['EP300']
    df = df[['chr', 'start', 'end', 'name']]
    a = pybedtools.BedTool(df.values.tolist())
    a = a.sequence(fi=in_fa, fo = out_fa,nameOnly=True)

def main() -> None:
    TSV_FOLDER = Path("Data/tsv")
    FASTA_FOLDER = Path("Data/fasta")
    GENOME_FASTA = Path("Data/hg38.fa")
    for file in os.listdir(TSV_FOLDER):
        if file.endswith(".tsv"):
            file = TSV_FOLDER / file
            start = datetime.now()
            print(f"Starting file: {file.name} at {start.strftime('%H:%M:%S.%f')}")
            output_fasta = FASTA_FOLDER / f"{file.stem}.fasta"
            tsv_to_fasta(file, GENOME_FASTA, output_fasta)
            print(f"\t\tTime Taken: {(datetime.now()-start).total_seconds()} seconds")
    return

if __name__ == "__main__":
    main()