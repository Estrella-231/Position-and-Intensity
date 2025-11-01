import h5py
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='data/TCIR-train.h5')
parser.add_argument('--val', default='data/TCIR-val.h5')
parser.add_argument('--n', type=int, default=200)
args = parser.parse_args()

train_path = Path(args.train)
val_path = Path(args.val)

if not train_path.exists():
    print('Train file not found:', train_path)
    raise SystemExit(1)

print(f'Reading {train_path}...')
# Read info via pandas
info = pd.read_hdf(str(train_path), key='info', mode='r')
print('info shape:', info.shape)

# Read matrix via h5py
with h5py.File(train_path, 'r') as f:
    matrix = f['matrix']
    total = matrix.shape[0]
    n = min(args.n, total)
    print(f'Total samples: {total}, extracting: {n}')

    # Create new h5
    with h5py.File(str(val_path), 'w') as out:
        # write matrix subset
        out.create_dataset('matrix', data=matrix[:n], compression='gzip')

# write info subset with pandas
info.iloc[:n].to_hdf(str(val_path), key='info')
print('Created', val_path)
