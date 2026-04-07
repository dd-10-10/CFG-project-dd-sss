from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import pdb
# Import only your vectorizer from your existing script
from svm_script import seq_vec 

def tune_w_linear(file, t_file, k):
    # pdb.set_trace()
    print("Extracting features (this only happens once)...")
    train_ids, train_vecs = seq_vec(file, k)
    test_ids, test_vecs = seq_vec(t_file, k)
    
    N_train, cols = train_vecs.shape
    d = cols - 1 # The number of k-mer dimensions (4**k)
    
    print("Pre-allocating mapped feature matrices...")
    # 1. Pre-allocate the new mapped feature matrices
    # Size is (N, d + 2) because we replace the 1 ATAC column with 2 mapped columns
    X_train = np.zeros((N_train, d + 2), dtype=np.float32)
    X_test = np.zeros((test_vecs.shape[0], d + 2), dtype=np.float32)
    
    # 2. Insert the scaled k-mer counts (this part is constant for all w)
    X_train[:, :d] = train_vecs[:, :-1] / np.sqrt(d)
    X_test[:, :d] = test_vecs[:, :-1] / np.sqrt(d)
    
    # Extract the binary ATAC/B-status variable for easy access
    b_train = train_vecs[:, -1]
    b_test = test_vecs[:, -1]

    w_arr = np.linspace(0, 2, 25)
    acc_arr = np.zeros_like(w_arr)
    
    print("Starting LinearSVC tuning loop...")
    for i, w in enumerate(w_arr):
        print(f"w={w:.10f}", end=': ', flush=True)
        
        # 3. Update only the last two columns dynamically based on the current w
        sqrt_w = np.sqrt(abs(w))
        
        X_train[:, -2] = sqrt_w * b_train
        X_train[:, -1] = sqrt_w * (1 - b_train)
        
        X_test[:, -2] = sqrt_w * b_test
        X_test[:, -1] = sqrt_w * (1 - b_test)
        
        # 4. Train with LinearSVC 
        # dual=False is highly optimized for when N_samples > N_features
        clf = LinearSVC(dual=False, max_iter=10000, C=0.0001)
        clf.fit(X_train, train_ids)
        print(clf.coef_[0, -2:], end=' ')
        
        # Predict and calculate accuracy
        preds = clf.predict(X_test)
        acc = (preds == test_ids).mean()
        
        acc_arr[i] = acc
        print(f"acc={acc:.8f}", flush=True)
        
    # Plotting
    plt.plot(w_arr, acc_arr)
    plt.xlabel('w')
    plt.ylabel('Accuracy')
    plt.title(f'SVM w Tuning (Linear Mapping) - k={k}')
    
    fig_path = Path(f"SVMScripts/tuning/w_tuning_linear_{file.stem}_{t_file.stem}_{k}.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    print(f"Plot saved to {fig_path}")

if __name__== "__main__":
    # Point these to your actual 90k sequence files
    file = Path("data/temp/chr4_200bp_bins_CTCF_5.fa")
    t_file = Path("data/temp/chr1_200bp_bins_CTCF_1.fa")
    k = 4
    tune_w_linear(file, t_file, k)