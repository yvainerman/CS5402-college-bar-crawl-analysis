import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "processed")

def load_accelerometer_data():
    path = os.path.join(DATA_DIR, "all_accelerometer_data_pids_13.csv")
    acc = pd.read_csv(path)

    acc["time"] = pd.to_datetime(acc["time"], unit="ms")

    acc["x"] = pd.to_numeric(acc["x"], errors="coerce")
    acc["y"] = pd.to_numeric(acc["y"], errors="coerce")
    acc["z"] = pd.to_numeric(acc["z"], errors="coerce")

    acc = acc.dropna()

    # hello yev
    # this is where i remove the big spikes in the data that are messing up the averages
    # remove really large spiking in data (sensor errors)
    acc = acc[(acc["x"].abs() < 50) &
              (acc["y"].abs() < 50) &
              (acc["z"].abs() < 50)]

    return acc

def load_pids():
    path = os.path.join(DATA_DIR, "pids.txt")
    with open(path) as f:
        return [line.strip() for line in f]

def load_phone_types():
    path = os.path.join(DATA_DIR, "phone_types.csv")
    df = pd.read_csv(path)
    return dict(zip(df.pid, df.phonetype))

def load_tac(pid):
    path = os.path.join(DATA_DIR, "clean_tac", f"{pid}_clean_TAC.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s") # time field is unix in s
    return df

WINDOW_SIZE = 400   # 10 sec at 40hz
STRIDE = 200        # overlap windows by halfway

def create_windows(df):
    values = df[["x", "y", "z"]].values
    times = df["time"].values

    windows = []
    window_times = []

    for i in range(0, len(values) - WINDOW_SIZE, STRIDE):
        w = values[i:i+WINDOW_SIZE]
        t = times[i + WINDOW_SIZE // 2]  # center timestamp

        windows.append(w)
        window_times.append(t)

    return windows, window_times

def label_windows(window_times, tac_df):
    tac_df = tac_df.set_index("timestamp").sort_index()

    # combine/interpolate
    combined_index = tac_df.index.union(pd.to_datetime(window_times))
    tac_interp = tac_df.reindex(combined_index).interpolate().loc[window_times]

    labels = (tac_interp["TAC_Reading"] >= 0.08).astype(int) # someone is considered intoxicated asf if tac >= 0.08
    return labels.values

def extract_features(window):
    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]

    mag = np.sqrt(x**2 + y**2 + z**2)

    return [
        x.mean(), y.mean(), z.mean(),
        x.std(), y.std(), z.std(),
        mag.mean(), mag.std()
    ]

def build_dataset():
    acc = load_accelerometer_data()
    pids = load_pids()
    phone_map = load_phone_types()

    print(acc.dtypes)
    print(acc[["x","y","z"]].head())
    print(acc[["x","y","z"]].describe())

    X = []
    y = []
    pid_list = []

    for pid in pids:
        print(f"processing {pid}...")

        # filter accelerometer data for this pid
        acc_pid = acc[acc["pid"] == pid].sort_values("time")

        # skip if not enough data
        if len(acc_pid) < WINDOW_SIZE:
            continue

        tac_df = load_tac(pid)

        # create windows
        windows, window_times = create_windows(acc_pid)

        # label them
        labels = label_windows(window_times, tac_df)

        # phone type feature
        phone_feature = 1 if phone_map[pid] == "iPhone" else 0

        # build dataset
        for w, label in zip(windows, labels):
            feats = extract_features(w)
            feats.append(phone_feature)

            X.append(feats)
            y.append(label)
            pid_list.append(pid)

    return np.array(X), np.array(y), np.array(pid_list)