import numpy as np
from preprocess import load_tab, load_seq


def report_tabular(X, y, pids):
    print("\nTABULAR DATA")
    print(f"X shape: {X.shape}   y shape: {y.shape}   pids shape: {pids.shape}")

    print("\nFeature summary (averaged across all windows):")
    print(f"  x_mean   : {np.mean(X[:, 0]):+.4f}")
    print(f"  y_mean   : {np.mean(X[:, 1]):+.4f}")
    print(f"  z_mean   : {np.mean(X[:, 2]):+.4f}")
    print(f"  x_std    : {np.mean(X[:, 3]):.4f}")
    print(f"  y_std    : {np.mean(X[:, 4]):.4f}")
    print(f"  z_std    : {np.mean(X[:, 5]):.4f}")
    print(f"  mag_mean : {np.mean(X[:, 6]):.4f}")
    print(f"  mag_std  : {np.mean(X[:, 7]):.4f}")

    print("\nPhone type distribution:")
    print(f"  iPhone (1): {np.sum(X[:, -1] == 1)}")
    print(f"  Android (0): {np.sum(X[:, -1] == 0)}")


def report_sequential(X, y, pids):
    print("\nSEQUENTIAL DATA")
    print(f"X shape: {X.shape}   y shape: {y.shape}   pids shape: {pids.shape}")

    print("\nChannel stats (x, y, z, magnitude):")
    names = ["x", "y", "z", "mag"]

    for i, name in enumerate(names):
        print(f"  {name}: mean={np.mean(X[..., i]):+.4f}  std={np.std(X[..., i]):.4f}")

def main():

    # load datasets from folders
    X_tab, y_tab, pids_tab = load_tab()
    X_seq, y_seq, pids_seq = load_seq()

    # safety check: they MUST match
    assert np.array_equal(y_tab, y_seq), "ERROR: tab/seq labels do not match"
    assert np.array_equal(pids_tab, pids_seq), "ERROR: tab/seq pids do not match"

    print("\nDATASET OVERVIEW")

    print(f"Total windows: {len(y_tab)}")
    print(f"Unique participants: {len(np.unique(pids_tab))}")

    # LABEL DISTRIBUTION
    print("\nLabel distribution:")
    print(f"  sober (0)      : {np.sum(y_tab == 0)}")
    print(f"  intoxicated (1): {np.sum(y_tab == 1)}")
    print(f"  drunk ratio    : {np.mean(y_tab):.4f}")

    # PER PERSON BREAKDOWN
    print("\nPer-participant breakdown:")
    for pid in sorted(np.unique(pids_tab)):
        mask = pids_tab == pid
        total = np.sum(mask)
        drunk = np.sum(y_tab[mask])

        print(f"  {pid}: total={total:5d}  drunk={drunk:5d} ({100*drunk/total:.1f}%)")

    # FEATURE REPORTS
    report_tabular(X_tab, y_tab, pids_tab)
    report_sequential(X_seq, y_seq, pids_seq)

    # ONE SAMPLE CHECK
    print("\nSAMPLE")
    print("Tabular sample:", X_tab[0])
    print("Label:", y_tab[0], "PID:", pids_tab[0])

    print("\nSequential sample shape:", X_seq[0].shape)
    print("Label:", y_seq[0], "PID:", pids_seq[0])

if __name__ == "__main__":
    main()