# Predicting TF binding with Markov Models
#### Suchet Samir Sadekar (20221233), Dheeraj Deshpande (20221090)
This is a program meant to calculate whether a given transcription factor will bind to test sequences, given training data. This is achieved using Markov models of specifiable order.

## Dependencies
### Versions
This code was written in **Python 3.12.12** and with the following libraries:
- Numpy 2.4.1
- Pandas 2.3.3
- Matplotlib 3.10.8
- Scikit-learn 1.8.0
- Pybedtools 0.12.0 (Bedtools 2.31.1)

The code may work in earlier versions, but we cannot guarantee functionality in such cases. It will definitely not work for Python versions earlier than Python 3.6.

### Genome file
This program requires the `hg38.fa` genome file to run. To download it, run the following command in terminal:\
`wget https://hgdownload.gi.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz -O data/hg38.fa.gz`\
Next, to unzip the file, run:\
`gunzip data/hg38.fa.gz`

## Usage
To run the program for chromosome 4, run the following command in terminal:\
`python pythonScripts/run_me.py data/tsv/chr4_200bp_bins.tsv path/to/output_dir <m> <k> <tf>`

For more information and optional arguments, run the following command in terminal:\
`python pythonScripts/run_me.py -h`

## Structure
- `data`
    - `Midsem_Eval`: files and directories for submission
    - `Output`: default output location
    - `temp`: intermediate storage location for .fa and .npy files
    - `tsv`: input tsv files
- `pythonScripts`
    - `run_me.py`: the main program, which accomplishes the stated task using the other scripts
    - `globals.py`: a file containing some useful variables for reference elsewhere
    - `tsv_to_fasta.py`: script to extract fasta files from input tsv files
    - `create_cv_folds.py`: script to create training/ validation folds from input data
    - `markov_mle.py`: script to train a markov model on training data
    - `markov_score.py`: script to calculate log-odds score for input testing data, based on markov mle arrays
