from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from svm_script import test

def test_w(file, t_file, k):
    w_arr= np.linspace(1e-3, 1, 25)
    acc_arr= np.zeros_like(w_arr)
    for i, w in enumerate(w_arr):
        print(w, end=' ', flush=True)
        acc = test(file, t_file, k, w)[0]
        acc_arr[i]= acc
        print(acc, flush=True)
    plt.plot(w_arr, acc_arr)
    fig_path = Path(f"SVMScripts/tuning/w_tuning_{file.stem}_{t_file.stem}_{k}.png")
    if not fig_path.parent.exists:
        fig_path.parent.mkdir()
    plt.savefig(fig_path)

if __name__== "__main__":
    file= Path("data/temp/chr22_200bp_bins_CTCF_1.fa")
    t_file= Path("data/temp/chr22_200bp_bins_CTCF_1.fa")
    k= 6
    test_w(file, t_file, k)
