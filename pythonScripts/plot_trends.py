from pathlib import Path
from datetime import datetime
import numpy as np
from numpy import polyfit
import matplotlib.pyplot as plt

def quadratic(x, a, b, c):
    return a*x*x + b*x + c
quadratic = np.vectorize(quadratic)

log_file = Path("data/MidSem_SuchetsRun_1/logs.txt")

logs = []
with open(log_file, 'r') as f:
    for line in f:
        if "pc" in line or "Memory" in line:
            logs.append(line.strip().replace('\t', ' ').replace('  ', ' '))
logs = np.array(logs).reshape(11, 3, 2)

# Time and Memory
time = np.zeros((3, 11))
memory = np.zeros((3, 11))

for m, temp in enumerate(logs):
    for tf, line in enumerate(temp):
        day = int(line[0][0:2])
        month = int(line[0][3:5])
        year = int(line[0][6:8])+2000
        hour = int(line[0][9:11])
        minute = int(line[0][12:14])
        second = int(line[0][15:17])
        start = datetime(year = year, month=month, day=day, hour=hour, minute=minute, second=second)
        
        day = int(line[1][0:2])
        month = int(line[1][3:5])
        year = int(line[1][6:8])+2000
        hour = int(line[1][9:11])
        minute = int(line[1][12:14])
        second = int(line[1][15:17])
        end = datetime(year = year, month=month, day=day, hour=hour, minute=minute, second=second)

        time[tf][m] = (end - start).total_seconds() # in seconds
        memory[tf][m] = int(line[1].split(' ')[-2])/(1024*1024) # in MB

X = np.linspace(0, 10, 1000)
plt.figure(0, figsize=(6,6))

plt.plot(range(11), time.T, label = ['CTCF', 'REST', 'EP300'])

avg_time_complexity = np.polyfit(range(11), time.mean(axis=0), deg=2)
plt.plot(X, quadratic(X, *avg_time_complexity), 'k--', label = f"$O(m^2)$ fit")

plt.title("Time Taken")
plt.xlabel("Order of the Markov Model")
plt.ylabel("Time (seconds)")
plt.legend()
plt.tight_layout()
plt.savefig(Path("data/MidSem_SuchetsRun_1/TimeTaken.png"))

plt.figure(1, figsize=(6,6))
plt.plot(range(11), memory.T, label = ['CTCF', 'REST', 'EP300'])
plt.title("Space Used")
plt.xlabel("Order of the Markov Model")
plt.ylabel("Memory (MB)")
plt.legend()
plt.tight_layout()
plt.savefig(Path("data/MidSem_SuchetsRun_1/MemoryUsage.png"))


# auROC and auPRC
roc = np.zeros((11, 3))
prc = np.zeros((11, 3))

files = [i for i in Path("data/MidSem_SuchetsRun_1/Output/").iterdir() if i.stem.startswith('A')]

for fname in files:
    m = int(fname.stem.split('_')[4][2:])
    tf = {'CTCF':0, "REST":1, 'EP300':2}[fname.stem.split('_')[-1][3:]]
    with open(fname, 'r') as f:
        data = f.readlines()
        roc[m][tf] = float(data[0].strip().split(' ')[-1])
        prc[m][tf] = float(data[1].strip().split(' ')[-1])

plt.figure(2, figsize=(6,6))
plt.plot(range(11), roc, label = ['CTCF', 'REST', 'EP300'])
plt.title("Average auROC across folds")
plt.xlabel("Order of the Markov Model")
plt.ylabel("auROC")
plt.legend()
plt.tight_layout()
plt.savefig(Path("data/MidSem_SuchetsRun_1/ROC_trend.png"))

plt.figure(3, figsize=(6,6))
plt.plot(range(11), prc, label = ['CTCF', 'REST', 'EP300'])
plt.title("Average auPRC across folds")
plt.xlabel("Order of the Markov Model")
plt.ylabel("auPRC")
plt.legend()
plt.tight_layout()
plt.savefig(Path("data/MidSem_SuchetsRun_1/PRC_trend.png"))
