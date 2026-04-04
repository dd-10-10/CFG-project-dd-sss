import numpy as np
import matplotlib.pyplot as plt

from svm_script import test

def test_w(file, t_file, k):
    w_arr= np.arange(0.5, 5, 0.5)
    acc_arr= np.zeros_like(w_arr)
    for i, w in enumerate(w_arr):
        acc, t_ids= test(file, t_file, k, w)
        acc_arr[i]= acc
    plt.plot(w_arr, acc_arr)
    plt.savefig(f"SVMScripts/tuning/w_tuning_{file}_{t_file}_{k}.png")

if __name__== "__main__":
    file= "data/temp/chr4_200bp_bins_CTCF_5.fa"
    t_file= "data/temp/chr4_200bp_bins_CTCF_5.fa"
    k= 2
    test_w(file, t_file, k)