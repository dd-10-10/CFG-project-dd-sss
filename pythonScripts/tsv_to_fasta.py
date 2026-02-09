from pathlib import Path
from datetime import datetime
import pybedtools
import os
import pandas as pd
# Convert TSV to FASTA

def tsv_to_fasta(in_tsv_path:Path, in_genome:Path) -> None:
    '''
    Convert a TSV file with genomic coordinates to a FASTA file using a reference genome.

    Arguments:
        in_tsv: Path to the input TSV file
        in_genome: Path to the genome FASTA file
    
    Returns:
        None
    '''
    out_fa = in_tsv_path.parent # Output Path
    if (out_fa / f"{in_tsv_path.stem}.fa").exists():
        print((out_fa / f"{in_tsv_path.stem}.fa"), "already exists, skipping")
        return
    df = pd.read_csv(in_tsv_path, sep="\t", dtype={'chr':str, 'start':int, 'end':int, "ATAC":str, 'CTCF':str, "REST":str, 'EP300':str})
    df['name'] = df['chr'] + ":" + df['start'].astype(str) + "-" + df['end'].astype(str) + "_" + df['ATAC'] + df['CTCF'] + df['REST'] + df['EP300']
    df = df[['chr', 'start', 'end', 'name']]
    a = pybedtools.BedTool(df.values.tolist())
    a = a.sequence(fi=in_genome, fo = out_fa / f"{in_tsv_path.stem}.fa",nameOnly=True)
    return

#TODO: Refactor
def main() -> None:
    TSV_FOLDER = Path("data/tsv")
    FASTA_FOLDER = Path("data/fasta")
    GENOME_FASTA = Path("data/hg38.fa")
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
