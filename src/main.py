import os
import numpy as np

from preprocess import build_dataset
from preprocess import BASE_DIR
from preprocess import DATA_DIR
from preprocess import CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)

USE_CACHE = False

# load or build the dataset
def load_or_build():
    x_path = os.path.join(CACHE_DIR, "X.npy")
    y_path = os.path.join(CACHE_DIR, "y.npy")
    p_path = os.path.join(CACHE_DIR, "pids.npy")

    if USE_CACHE and os.path.exists(x_path):
        print("loading cached dataset...")
        X = np.load(x_path)
        y = np.load(y_path)
        pids = np.load(p_path)
        return X, y, pids

    print("building dataset (this will take 10000 years)")
    X, y, pids = build_dataset()

    print("Saving dataset...")
    np.save(x_path, X)
    np.save(y_path, y)
    np.save(p_path, pids)

    return X, y, pids

def main():
    X, y, pids = load_or_build()

    print(f"Dataset shape: {X.shape}")
    
    print("\nLabel distribution:")
    print("0 (sober):", np.sum(y == 0))
    print("1 (intoxicated):", np.sum(y == 1))
    print("ratio drunk:", np.mean(y))
    
    print("\nFeature means:")
    print("x_mean:", np.mean(X[:, 0]))
    print("y_mean:", np.mean(X[:, 1]))
    print("z_mean:", np.mean(X[:, 2]))

    print("\nStd averages:")
    print("x_std:", np.mean(X[:, 3]))
    print("y_std:", np.mean(X[:, 4]))
    print("z_std:", np.mean(X[:, 5]))

    print("\nMagnitude mean:", np.mean(X[:, 6]))
    print("Magnitude std:", np.mean(X[:, 7]))

    print("\nPhone type distribution:")
    print("iPhone (1):", np.sum(X[:, -1] == 1))
    print("Android (0):", np.sum(X[:, -1] == 0))

    print("\nNumber of unique participants:", len(np.unique(pids)))

    print("\nSample feature row:")
    print(X[0])
    print("Label:", y[0], "PID:", pids[0])

if __name__ == "__main__":
    main()
