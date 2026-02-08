from pathlib import Path
import pandas as pd
import random

def cross_validate(in_tsv_path:Path, out_dir_path:Path, k:int, tf:str, random_state:int|None=None) -> None:
    '''
    Create n-folds for cross-validation of the input TSV file and saves the folds to a specified directory.

    Arguments:
        in_tsv_path: Path to the input TSV file
        out_dir_path: Path to the output directory for fold files
        k: Number of folds for cross-validation
        tf: Transcription factor to consider
        random_state: Random seed for reproducibility (default: None)
    Returns:
        None
    '''
    df = pd.read_csv(in_tsv_path, sep="\t", dtype={'chr':str, 'start':int, 'end':int, "ATAC":str, 'CTCF':str, "REST":str, 'EP300':str})
    df_bound = df[df[tf] == 'B'].reset_index(drop=True)
    len_b = len(df_bound)
    df_unbound = df[df[tf] == 'U'].reset_index(drop=True)
    len_u = len(df_unbound)
    if random_state is not None:
        random.seed(random_state)

    fold_size_b = len_b // k
    fold_size_u = len_u // k
    indices_b = list(range(k))*fold_size_b + list(range(k))[:len_b % k]
    indices_u = list(range(k))*fold_size_u + list(range(k))[:len_u % k]
    indices_b = random.sample(indices_b, len_b)
    indices_u = random.sample(indices_u, len_u)
    df_bound['fold'] = indices_b
    df_unbound['fold'] = indices_u

    names= []
    for fold in range(k):
        fold_b = df_bound[df_bound['fold'] == fold].drop(columns=['fold']).reset_index(drop=True)
        fold_u = df_unbound[df_unbound['fold'] == fold].drop(columns=['fold']).reset_index(drop=True)
        fold_df = pd.concat([fold_b, fold_u])
        fold_file = out_dir_path / f"{in_tsv_path.stem}_{tf}_{fold+1}.tsv"
        fold_df.to_csv(fold_file, sep="\t", index=False)
        names.append(fold_file.name)

    return names

def main() -> None:
    INPUT_TSV = Path("data/tsv/chr22_200bp_bins.tsv")
    OUTPUT_DIR = Path("data/cv")
    K_FOLDS = 5
    TF = "CTCF"
    RANDOM_STATE = 42
    cross_validate(INPUT_TSV, OUTPUT_DIR, K_FOLDS, TF, RANDOM_STATE)
    return

if __name__ == "__main__":
    main()
