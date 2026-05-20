import numpy as np
from pathlib import Path

gateman_dir = "./modules/dataset_module/datasets/GatemanJamming"
test_file = Path(gateman_dir) / "GPSL1+AMtone" / "GPSL1@20MSps-16bit.bin"

# Read first 20 samples both ways
with open(test_file, "rb") as f:
    raw_bytes = f.read(80)  # 20 complex samples * 4 bytes each

big    = np.frombuffer(raw_bytes, dtype=np.dtype(">i2"))  # big-endian
little = np.frombuffer(raw_bytes, dtype=np.int16)          # little-endian (native)

print("Big-endian first 10 I/Q pairs:")
print(big[:20].reshape(-1, 2))
print(f"  Range: [{big.min()}, {big.max()}]")

print("\nLittle-endian first 10 I/Q pairs:")
print(little[:20].reshape(-1, 2))
print(f"  Range: [{little.min()}, {little.max()}]")

# Quick power check across first 10000 samples
with open(test_file, "rb") as f:
    chunk = f.read(40000)

big_all    = np.frombuffer(chunk, dtype=np.dtype(">i2")).reshape(-1, 2)
little_all = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 2)

big_power    = np.mean(big_all[:, 0]**2 + big_all[:, 1]**2)
little_power = np.mean(little_all[:, 0]**2 + little_all[:, 1]**2)

print(f"\nMean power (big-endian):    {big_power:.1f}")
print(f"Mean power (little-endian): {little_power:.1f}")