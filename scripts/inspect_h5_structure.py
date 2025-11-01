import h5py
import pandas as pd
import sys
from pathlib import Path

train_path = Path("data") / "TCIR-train.h5"
if not train_path.exists():
    print(f"File not found: {train_path}")
    sys.exit(1)

print(f"Inspecting: {train_path}\n")
with h5py.File(train_path, 'r') as f:
    def recurse(name, obj):
        print(name, type(obj))
        if isinstance(obj, h5py.Dataset):
            print("  shape:", obj.shape, "dtype:", obj.dtype)
    f.visititems(recurse)

# Try to read 'info' via pandas (may require pytables)
try:
    info = pd.read_hdf(str(train_path), key='info', mode='r')
    print('\nPandas read_hdf succeeded. info shape:', info.shape)
    print(info.head())
except Exception as e:
    print('\nPandas read_hdf failed:', repr(e))
