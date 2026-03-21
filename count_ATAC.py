import os
import pandas as pd
from pathlib import Path
from itertools import product

FOLDER = Path('data/tsv')

dir = {"".join(i):0 for i in product(['U', 'B'], ['U', 'B'], ['U', 'B'], ['U', 'B'])}
for file in [file for file in os.listdir(FOLDER) if file.endswith('bins.tsv')]:
    df = pd.read_csv(FOLDER / file, sep='\t')
    t = df[['ATAC', 'CTCF', 'REST', 'EP300']].values
    for ind in t:
        dir["".join(ind)]+=1
    print(file, '\n', dir.items())

print(dir.items())