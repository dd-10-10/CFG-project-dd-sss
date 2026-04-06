from pathlib import Path
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Bio import SeqIO

from globals import base_dict, BATCH_SIZE, device

logging.basicConfig(filename='NNScripts/progress.log', filemode='a',
                    format='%(asctime)s\t%(message)s', level=logging.INFO)
class DenseNetwork(nn.Module):
    def __init__(self):
        super(DenseNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(200*4, 400), # ATAC not accounted for
            nn.ReLU(),
            nn.Linear(400, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.convLayers = nn.Sequential(
            nn.Conv1d(5, 20, 11, 5), # 5th channel for ATAC
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(20, 100, 7, 3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        self.linearLayer = nn.Sequential(
            nn.Linear(200, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.convLayers(x)
        x = self.flatten(x)
        x = self.linearLayer(x)
        return x

class FastaMapDataset(Dataset):
    def __init__(self, records):
        self.records = records
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        def one_hot(seq: str) -> np.ndarray:
            vec = np.zeros((len(seq), 4))
            for idx, val in enumerate(seq):
                try:
                    vec[idx][base_dict[val]] = 1.0
                except KeyError:
                    pass
            return vec

        record = self.records[idx]
        
        seq = torch.tensor(one_hot(record.seq), dtype=torch.float32)
        ids = torch.tensor([1 if i == 'B' else 0 for i in record.id[-3:]], dtype=torch.float32)
        atac = torch.full((200, 1), 1 if record.id[-4] == 'B' else 0, dtype=torch.float32)
        
        seq = torch.concat([seq, atac], dim=1).transpose(0, 1)
        return seq, ids

def load_data(fasta_path:Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    all_records = list(SeqIO.parse(fasta_path, 'fasta'))
    train_records, temp_records = train_test_split(all_records, test_size=0.3, random_state=42)
    val_records, test_records = train_test_split(temp_records, test_size=0.5, random_state=42)
    
    train_dataset = FastaMapDataset(train_records)
    val_dataset = FastaMapDataset(val_records)
    test_dataset = FastaMapDataset(test_records)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train(net: ConvNetwork | DenseNetwork, data_loaders: tuple) -> ConvNetwork | DenseNetwork:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    train_loader, val_loader, test_loader = data_loaders
    logging.info("Epoch\tTrain_Loss\tVal_Loss")
    
    for epoch in range(10):
        print(f"Starting Epoch {epoch+1}")
        net.train()
        running_train_loss = 0.0
        batch = 0
        for batch, (seq, ids) in enumerate(train_loader):
            seq = seq.to(device)
            ids = ids.to(device)
            
            optimizer.zero_grad()
            outputs = net(seq)
            train_loss = criterion(outputs, ids)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        running_train_loss /= (batch+1)

        net.eval()
        batch = 0
        running_val_loss = 0.0
        for batch, (seq, ids) in enumerate(test_loader):
            seq = seq.to(device)
            ids = ids.to(device)
            outputs = net(seq)
            val_loss = criterion(outputs, ids)
            running_val_loss += val_loss.item()
        running_val_loss /= (batch+1)

        if epoch % 1 == 0:
            logging.info(f"{epoch+1:<3}:\t{running_train_loss}\t{running_val_loss}")
    
    net.eval()
    running_test_loss = 0.0
    batch = 0
    for batch, (seq, ids) in enumerate(val_loader):
        seq = seq.to(device)
        ids = ids.to(device)
        outputs = net(seq)
        test_loss = criterion(outputs, ids)
        running_test_loss += test_loss.item()
    running_test_loss /= (batch + 1)
    print(f'\nTest Loss={running_test_loss}')
    
    return net

def main() -> None:
    print("Building Data Structure")
    fasta_path = Path("data/tsv/chrAll.fa")
    train_loader, val_loader, test_loader = load_data(fasta_path)
    
    print("Start Training")
    net = ConvNetwork().to(device)
    net = train(net, (train_loader, val_loader, test_loader))
    
    torch.save(net.state_dict(), "NNScripts/model1.pth")
    return

if __name__ == '__main__':
    main()
