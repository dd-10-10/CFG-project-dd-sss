from pathlib import Path
base_dict = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
tf_dict= {"ATAC": 0, "CTCF": 1, "REST":2, "EP300":3}
genome_path = Path("/home/dheeraj/cfg/wk2/hg38.fa")

def seq_to_pos(seq: str) -> int:
    '''
    Takes a sequence as input, interprets it as a base-4 number, and returns the number in base-10.
    
    Argument:
        seq: Input sequence
    
    Output:
        Representation of seq in base-10
    '''
    pos= 0
    for i,v in enumerate(seq[::-1]):
        pos+= base_dict[v]*(4**i)
    return pos
