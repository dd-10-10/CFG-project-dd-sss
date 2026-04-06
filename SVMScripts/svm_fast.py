from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from svm_script import seq_vec

def prep(file, t_file, k):
    ids, vecs= seq_vec(file, k)
    t_ids, t_vecs= seq_vec(t_file, k)

    l= vecs.shape[1]- 1

    base_dot= np.dot(vecs[:, :-1], vecs[:, :-1].T)/ l
    base_atac= 1- abs(vecs[:, -1]- vecs[:, -1])
    test_dot= np.dot(t_vecs[:, :-1], vecs[:, :-1].T)/ l
    test_atac= 1- abs(t_vecs[:, -1]- vecs[:, -1])

    return ids, t_ids, base_dot, base_atac, test_dot, test_atac

def tune_w(file, t_file, k, w_arr):
    ids, t_ids, base_dot, base_atac, test_dot, test_atac= prep(file, t_file, k)
    
    acc_arr= np.zeros_like(w_arr)
    for i, w in enumerate(w_arr):

        base_ker= base_dot + w * base_atac
        test_ker= test_dot + w * test_atac

        mmc= svm.SVC(kernel= "precomputed")
        mmc.fit(base_ker, ids)
        t_pred= mmc.predict(test_ker)

        match= t_pred== t_ids
        acc= match.sum()/len(match)
        acc_arr[i]= acc
        
        print(f"w= {w}, acc= {acc}")
    
    plt.plot(w_arr, acc_arr)
    plt.xlabel("w")
    plt.ylabel("Accuracy")
    fig_path = Path(f"SVMScripts/tuning/w_tuning_{file.stem}_{t_file.stem}_{k}.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)

if __name__== "__main__":
    file= Path("data/temp/chr4_200bp_bins_CTCF_1.fa")
    t_file= Path("data/temp/chr4_200bp_bins_CTCF_1.fa")
    k= 6
    w_arr= np.linspace(1e-3, 1, 25)
    tune_w(file, t_file, k, w_arr)