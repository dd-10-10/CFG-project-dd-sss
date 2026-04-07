import time
import itertools
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

def get_all_classes(file_path):
    '''
    Lightning-fast pre-scan to find all unique labels in the FASTA headers.
    Ensures SGDClassifier knows exactly what classes to expect.
    '''
    unique_classes = set()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip()[1:]
                unique_classes.add(header[-3])
    return np.array(list(unique_classes))

def fasta_chunk_generator(file_path, tf, atac, chunk_size=100000):
    # ... (Keep this exactly the same as the previous version) ...
    ids, seqs, b_flags = [], [], []
    
    with open(file_path, 'r') as f:
        header, seq_acc = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_acc)
                    if "n" not in seq.lower(): 
                        ids.append(header[tf])
                        b_flags.append(1 if len(header) >= 4 and header[atac] == "B" else 0)
                        seqs.append(seq)
                        
                        if len(seqs) >= chunk_size:
                            yield np.array(ids), seqs, np.array(b_flags).reshape(-1, 1).astype(np.uint8)
                            ids, seqs, b_flags = [], [], []
                            
                header = line[1:] 
                seq_acc = []
            else:
                seq_acc.append(line)
        
        if header is not None:
            seq = "".join(seq_acc)
            if "n" not in seq.lower():
                ids.append(header[tf])
                b_flags.append(1 if len(header) >= 4 and header[atac] == "B" else 0)
                seqs.append(seq)
                if seqs:
                    yield np.array(ids), seqs, np.array(b_flags).reshape(-1, 1).astype(np.uint8)

def chunked_kmer_svm(train_file, test_file, k):
    # 1. Setup Fixed Vocabulary & Vectorizer
    print(f"{time.time():.2f} - Initializing setup...")
    vocab = [''.join(p) for p in itertools.product('acgt', repeat=k)]
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), vocabulary=vocab, lowercase=True)
    
    # 2. Dynamically Detect Classes & Setup SGDClassifier
    print(f"{time.time():.2f} - Pre-scanning headers for unique classes...")
    all_classes = get_all_classes(train_file)
    print(f"   -> Detected classes: {all_classes}")
    
    clf = SGDClassifier(loss='hinge', tol=1e-3, random_state=42, n_jobs=-1, class_weight= {"B":3608612/(22817*2), "U": 3608612/((3608612-22817)*2)})
    if Path("ep300_model.pkl").exists():
        print(f"{time.time():.2f} - Loading Pre-trained model...")
        with open("ep300_model.pkl", "rb") as model_file:
            clf= pkl.load(model_file)
    else:
        # 3. Train in Chunks
        print(f"{time.time():.2f} - Beginning chunked training...")
        chunk_num = 1
        for y_chunk, seqs_chunk, b_chunk in fasta_chunk_generator(train_file, -1, -4, chunk_size=150000):
            X_kmers_chunk = vectorizer.transform(seqs_chunk)
            X_dense_chunk = np.hstack([X_kmers_chunk.toarray().astype(np.uint8), b_chunk])
            
            clf.partial_fit(X_dense_chunk, y_chunk, classes=all_classes)
            print(f"   -> Trained on chunk {chunk_num}...")
            chunk_num += 1

        print(f"{time.time():.2f} - Training complete. Beginning chunked evaluation...")

    # 4. Evaluate in Chunks
    correct_predictions = 0
    total_predictions = 0
    conf_mat= np.zeros((2,2))
    with open("ep300_model.pkl", "wb") as model_file:
      pkl.dump(clf, model_file)
    with open("preds.txt", "w") as out_file:
      for y_test_chunk, seqs_test_chunk, b_test_chunk in fasta_chunk_generator(test_file, -2, -1, chunk_size=150000):
          X_kmers_test = vectorizer.transform(seqs_test_chunk)
          X_dense_test = np.hstack([X_kmers_test.toarray().astype(np.uint8), b_test_chunk])
          
          preds = clf.predict(X_dense_test)
          out_file.writelines([str(i) for i in preds] + ['\n'])
          conf_mat += confusion_matrix(preds, y_test_chunk, labels= ["U", "B"])
          correct_predictions += (preds == y_test_chunk).sum()
          total_predictions += len(y_test_chunk)

    accuracy = correct_predictions / total_predictions
    return accuracy, conf_mat

if __name__== "__main__":
    file = "chrAll.fa"
    t_file = "chrAll_unknown.fa"
    k = 4
    
    print(f"{time.time():.2f} - Start")
    acc, conf_mat = chunked_kmer_svm(file, t_file, k)
    print(f"\nFinal Accuracy: {acc:.4f}")
    print(conf_mat)
    print(f"{time.time():.2f} - End")