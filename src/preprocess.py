import os
import numpy as np
import pandas as pd

# PATHS
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "project", "data")
CACHE_DIR  = os.path.join(BASE_DIR, "project", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "project", "models")

TAB_DIR = os.path.join(CACHE_DIR, "tab")
SEQ_DIR = os.path.join(CACHE_DIR, "seq")

# WINDOWING
WINDOW_SIZE = 400 # 400 samples = 10 seconds at 40 Hz
STRIDE      = 200 # 50% overlap, set to 400 for no overlap

# DATA LOADER
def load_accelerometer_data():
    path = os.path.join(DATA_DIR, "all_accelerometer_data_pids_13.csv")
    acc = pd.read_csv(path)

    # convert time + numeric cleanup
    acc["time"] = pd.to_datetime(acc["time"], unit="ms")

    acc["x"] = pd.to_numeric(acc["x"], errors="coerce")
    acc["y"] = pd.to_numeric(acc["y"], errors="coerce")
    acc["z"] = pd.to_numeric(acc["z"], errors="coerce")

    acc = acc.dropna()

    # remove sensor spikes
    acc = acc[
        (acc["x"].abs() < 50) &
        (acc["y"].abs() < 50) &
        (acc["z"].abs() < 50)
    ]

    return acc

# participant ids
def load_pids():
    path = os.path.join(DATA_DIR, "pids.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

# phone types
def load_phone_types():
    path = os.path.join(DATA_DIR, "phone_types.csv")
    df = pd.read_csv(path)
    return dict(zip(df.pid, df.phonetype))

def load_tac(pid):
    path = os.path.join(DATA_DIR, "clean_tac", f"{pid}_clean_TAC.csv")
    df = pd.read_csv(path)
    # convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    # remove dupe timestamps
    df = df.drop_duplicates(subset="timestamp")
    # sort by time.
    df = df.sort_values("timestamp")
    return df

# WINDOWING + LABELING
def create_windows(df):
    values = df[["x", "y", "z"]].values
    times = df["time"].values

    windows = []
    window_times = []

    for i in range(0, len(values) - WINDOW_SIZE, STRIDE):
        windows.append(values[i:i + WINDOW_SIZE])
        window_times.append(times[i + WINDOW_SIZE // 2])

    return windows, window_times


def label_windows(window_times, tac_df):
    tac_df = tac_df.set_index("timestamp")

    # align timestamps
    all_times = tac_df.index.union(pd.DatetimeIndex(window_times))

    tac_interp = (
        tac_df.reindex(all_times)
              .interpolate(method="time")
              .ffill()
              .bfill()
              .loc[window_times]
    )
    
    # legal limit threshold
    labels = (tac_interp["TAC_Reading"] >= 0.08).astype(int)
    return labels.values
    
# FEAT EXTRACTION FOR TABULAR
def extract_features(window):
    x, y, z = window[:, 0], window[:, 1], window[:, 2]

    mag = np.sqrt(x**2 + y**2 + z**2)

    return [
        x.mean(), y.mean(), z.mean(),
        x.std(),  y.std(),  z.std(),
        mag.mean(), mag.std()
    ]

# SAVE
def build_and_save():
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(SEQ_DIR, exist_ok=True)

    acc = load_accelerometer_data()
    pids = load_pids()
    phone_map = load_phone_types()

    tab_X, tab_y, tab_pids = [], [], []
    seq_X, seq_y, seq_pids = [], [], []

    for pid in pids:
        print(f"processing {pid}...")

        acc_pid = acc[acc["pid"] == pid].sort_values("time")

        if len(acc_pid) < WINDOW_SIZE:
            continue

        try:
            tac_df = load_tac(pid)
        except FileNotFoundError:
            continue

        windows, window_times = create_windows(acc_pid)
        labels = label_windows(window_times, tac_df)

        phone_feature = 1 if phone_map.get(pid) == "iPhone" else 0

        for w, label in zip(windows, labels):

            # tabular data
            feats = extract_features(w)
            feats.append(phone_feature)

            tab_X.append(feats)
            tab_y.append(label)
            tab_pids.append(pid)

            # sequential data
            mag = np.sqrt((w * w).sum(axis=1, keepdims=True))
            seq = np.concatenate([w, mag], axis=1)

            seq_X.append(seq)
            seq_y.append(label)
            seq_pids.append(pid)

    # convert + save TABULAR
    np.save(os.path.join(TAB_DIR, "X.npy"), np.array(tab_X, dtype=np.float32))
    np.save(os.path.join(TAB_DIR, "y.npy"), np.array(tab_y))
    np.save(os.path.join(TAB_DIR, "pids.npy"), np.array(tab_pids))

    # convert + save SEQ
    np.save(os.path.join(SEQ_DIR, "X.npy"), np.array(seq_X, dtype=np.float32))
    np.save(os.path.join(SEQ_DIR, "y.npy"), np.array(seq_y))
    np.save(os.path.join(SEQ_DIR, "pids.npy"), np.array(seq_pids))

    print("\nDONE")
    print("Tab:", len(tab_y))
    print("Seq:", len(seq_y))


# LOADERS FOR TRAIN/TEST SCRIPT
def load_tab():
    return (
        np.load(os.path.join(TAB_DIR, "X.npy")),
        np.load(os.path.join(TAB_DIR, "y.npy")),
        np.load(os.path.join(TAB_DIR, "pids.npy")),
    )


def load_seq():
    return (
        np.load(os.path.join(SEQ_DIR, "X.npy")),
        np.load(os.path.join(SEQ_DIR, "y.npy")),
        np.load(os.path.join(SEQ_DIR, "pids.npy")),
    )

if __name__ == "__main__":
    build_and_save()