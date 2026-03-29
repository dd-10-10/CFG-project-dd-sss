from pathlib import Path
import numpy as np
import torch
from torch import nn
from Bio import SeqIO
from globals import base_dict

class DenseNetwork(nn.Module):
    def __init__(self):
        super(DenseNetwork, self).__init__()
        self.layer1 = nn.Linear(200*4, 400)
        self.layer2 = nn.Linear(400, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.sigmoid(x)
        return x

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv1d(4, 20, 3, 1)
        self.conv2 = nn.Conv1d(20, 100, 3, 1)
        self.linear = nn.Linear(2100, 3)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.celu(x)
        x = nn.functional.max_pool1d(x, 3)
        x = self.conv2(x)
        x = nn.functional.gelu(x)
        x = nn.functional.max_pool1d(x, 3)
        x = self.flatten(x)
        x = self.linear(x)
        x = nn.functional.sigmoid(x)
        return x

def one_hot(seq: str) -> np.ndarray:
    vec = np.zeros((len(seq), 4))
    for idx, val in enumerate(seq):
        vec[idx][base_dict[val]] = 1.0
    return vec

def read_data(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    seqs = []
    ids = []
    for record in SeqIO.parse(filename, 'fasta'):
        seqs.append(one_hot(record.seq))
        ids.append([1 if i=='B' else 0 for i in record.id[-3:]])
    seqs = np.array(seqs)
    ids = np.array(ids)
    return seqs, ids

def train(net: ConvNetwork | DenseNetwork, x: torch.Tensor, y:torch.Tensor) -> ConvNetwork | DenseNetwork:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        #if epoch % 100 == 0:
        print(f"{epoch:<6}: {loss.item()}")
    return net

def main() -> None:
    filename = Path("data/tsv/chr11_200bp_bins.fa")
    seqs, ids = read_data(filename)
    seqs = torch.tensor(seqs, dtype=torch.float32).reshape(-1, 4, 200).to(device='cuda')
    ids = torch.tensor(ids, dtype=torch.float32).reshape(-1, 3).to(device='cuda')

    print(seqs.shape)
    print(ids.shape)
    net = ConvNetwork().to(device = 'cuda')
    net = train(net, seqs, ids)
    #torch.save(net.state_dict(), "NNScripts/model1.pth")
    return

main()
