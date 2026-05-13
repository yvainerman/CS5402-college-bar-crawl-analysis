import os
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

# PATHS
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "project", "data")
CACHE_DIR = os.path.join(BASE_DIR, "project", "processed")

TAB_DIR = os.path.join(CACHE_DIR, "tab")
SEQ_DIR = os.path.join(CACHE_DIR, "seq")

RESULTS_DIR = os.path.join(BASE_DIR, "project", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# CONSTANTS
# target sample rate (everyone is resampled to this)
SAMPLE_RATE = 40

# 10-second windows at 40 Hz
WINDOW_SIZE = 400

TAB_STRIDE = 400
SEQ_STRIDE = 40

MAX_TAC_GAP_MINUTES = 15

# kills sensor-glitch values
MAX_ABS_ACCEL = 19.6

# if gap between samples exceeds this time, split recording into sep
# cont segments so windows don't span a giant gap.
MAX_SENSOR_GAP_SECONDS = 5.0

# min window variation to keep window (drop flat windows)
MIN_MOTION_ENERGY = 1e-4

def load_accelerometer_data():

    path = os.path.join(
        DATA_DIR,
        "all_accelerometer_data_pids_13.csv"
    )

    acc = pd.read_csv(path)

    # remove invalid timestamps
    # many rows have time=0 placeholders
    acc = acc[acc["time"] > 0]

    # convert timestamps
    acc["time"] = pd.to_datetime(
        acc["time"],
        unit="ms"
    )

    # numeric conversion
    for col in ["x", "y", "z"]:
        acc[col] = pd.to_numeric(
            acc[col],
            errors="coerce"
        )

    # remove invalid rows
    acc = acc.dropna()

    # remove physically impossible cals
    acc = acc[
        (acc["x"].abs() < MAX_ABS_ACCEL) &
        (acc["y"].abs() < MAX_ABS_ACCEL) &
        (acc["z"].abs() < MAX_ABS_ACCEL)
    ]

    # remove duplicate timestamps within a participant
    acc = acc.drop_duplicates(
        subset=["pid", "time"]
    )

    return acc


def load_pids():

    path = os.path.join(DATA_DIR, "pids.txt")

    pids = []

    with open(path) as f:

        for line in f:

            pid = line.strip()

            if pid:
                pids.append(pid)

    return pids


def load_phone_types():

    path = os.path.join(
        DATA_DIR,
        "phone_types.csv"
    )

    df = pd.read_csv(path)

    phone_map = {}

    for i in range(len(df)):
        phone_map[df["pid"][i]] = df["phonetype"][i]

    return phone_map


def load_tac(pid):

    path = os.path.join(
        DATA_DIR,
        "clean_tac",
        f"{pid}_clean_TAC.csv"
    )

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s"
    )

    df = df.drop_duplicates(
        subset="timestamp"
    )

    df = df.sort_values("timestamp")

    return df


# segment recordings
def split_into_continuous_segments(df):
    df = df.sort_values("time").copy()

    diffs = (
        df["time"]
        .diff()
        .dt.total_seconds()
    )

    segment_breaks = diffs > MAX_SENSOR_GAP_SECONDS

    segment_ids = segment_breaks.cumsum()

    segments = []

    for _, segment in df.groupby(segment_ids):

        # need enough points to be useful even before resampling
        if len(segment) >= 2:
            segments.append(segment.reset_index(drop=True))

    return segments


# resampling
def resample_segment_to_uniform(segment, target_hz=SAMPLE_RATE):
    times = segment["time"].values

    # nanoseconds since epoch
    t_ns = times.astype("datetime64[ns]").astype(np.int64)

    t_start = t_ns[0]
    t_end = t_ns[-1]

    duration_seconds = (t_end - t_start) / 1e9

    if duration_seconds < (WINDOW_SIZE / target_hz):
        return None

    # uniform grid at target_hz
    step_ns = int(1e9 / target_hz)

    new_t_ns = np.arange(
        t_start,
        t_end + 1,
        step_ns,
        dtype=np.int64
    )

    if len(new_t_ns) < WINDOW_SIZE:
        return None

    # interpolate each axis
    x_new = np.interp(new_t_ns, t_ns, segment["x"].values)
    y_new = np.interp(new_t_ns, t_ns, segment["y"].values)
    z_new = np.interp(new_t_ns, t_ns, segment["z"].values)

    return pd.DataFrame({
        "time": pd.to_datetime(new_t_ns, unit="ns"),
        "x":    x_new.astype(np.float32),
        "y":    y_new.astype(np.float32),
        "z":    z_new.astype(np.float32),
    })


# windowing
def create_windows(df, stride):
    values = df[["x", "y", "z"]].values.astype(np.float32)

    times = df["time"].values

    windows = []

    start_times = []
    mid_times = []
    end_times = []

    i = 0

    while i + WINDOW_SIZE <= len(values):

        window = values[i : i + WINDOW_SIZE]

        start = times[i]
        end   = times[i + WINDOW_SIZE - 1]

        midpoint = times[i + WINDOW_SIZE // 2]

        # drop windows where phone is still
        per_axis_std = window.std(axis=0)

        if per_axis_std.max() < MIN_MOTION_ENERGY:

            i += stride
            continue

        windows.append(window)

        start_times.append(start)
        mid_times.append(midpoint)
        end_times.append(end)

        i += stride

    return windows, start_times, mid_times, end_times


# TAC labels
def label_windows(window_times,
                  tac_df,
                  threshold=0.08,
                  max_gap_minutes=15):

    tac_df = tac_df.sort_values("timestamp")

    tac_times_ns = (
        tac_df["timestamp"]
        .values
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    tac_vals = tac_df["TAC_Reading"].values.astype(np.float64)

    labels = []
    interpolated = []

    win_times_ns = (
        pd.to_datetime(window_times)
        .values
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    for t_ns in win_times_ns:

        # dist to nearest TAC reading
        deltas_ns = np.abs(tac_times_ns - t_ns)

        nearest_minutes = deltas_ns.min() / (60.0 * 1e9)

        # too far from any real TAC measurement
        if nearest_minutes > max_gap_minutes:

            labels.append(-1)
            interpolated.append(np.nan)

            continue

        # linear interp for TAC
        interp = np.interp(
            t_ns,
            tac_times_ns,
            tac_vals
        )

        interpolated.append(interp)

        if interp >= threshold:
            labels.append(1)
        else:
            labels.append(0)

    return (
        np.array(labels),
        np.array(interpolated, dtype=np.float64)
    )


# features
def zero_crossing_rate(signal):

    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - np.mean(signal)

    crossings = 0

    for i in range(1, len(signal)):

        prev_pos = signal[i - 1] >= 0
        curr_pos = signal[i] >= 0

        if prev_pos != curr_pos:
            crossings += 1

    return crossings / len(signal)


def dominant_frequency(signal):

    signal = np.asarray(signal, dtype=np.float32)

    signal = np.nan_to_num(signal)

    signal = signal - np.mean(signal)

    fft_mag = np.abs(np.fft.rfft(signal))

    freqs = np.fft.rfftfreq(
        len(signal),
        d=1.0 / SAMPLE_RATE
    )

    fft_mag[0] = 0

    idx = np.argmax(fft_mag)

    return freqs[idx]


def summarise_signal(signal):

    signal = np.asarray(signal, dtype=np.float32)

    return [

        np.mean(signal),
        np.std(signal),

        np.min(signal),
        np.max(signal),

        np.max(signal) - np.min(signal),

        zero_crossing_rate(signal),

        dominant_frequency(signal)
    ]


def extract_features(window, phone_feature):

    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]

    features = []

    for signal in [x, y, z]:

        features.extend(summarise_signal(signal))

    features.append(phone_feature)

    return features


# build data
def build_and_save():

    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(SEQ_DIR, exist_ok=True)

    acc = load_accelerometer_data()

    pids = load_pids()

    phone_map = load_phone_types()

    tab_X = []
    tab_y = []
    tab_pids = []

    seq_X = []
    seq_y = []
    seq_pids = []

    for pid in pids:

        print(f"PROCESSING {pid}")

        acc_pid = acc[
            acc["pid"] == pid
        ].sort_values("time")

        if len(acc_pid) < WINDOW_SIZE:

            print("Skipping, not enough samples.")

            continue

        try:
            tac_df = load_tac(pid)

        except FileNotFoundError:

            print("Skipping, no TAC.")

            continue

        diffs = acc_pid["time"].diff().dt.total_seconds()
        median_dt = diffs.median()
        est_hz = 1.0 / median_dt if median_dt and median_dt > 0 else float("nan")
        
        print(
            f"ACC range: "
            f"{acc_pid['time'].min()} -> "
            f"{acc_pid['time'].max()}  "
            f"(n={len(acc_pid):,}, ~{est_hz:.1f} Hz raw)"
        )

        print(
            f"TAC range: "
            f"{tac_df['timestamp'].min()} -> "
            f"{tac_df['timestamp'].max()}"
        )

        phone_feature = (
            1 if phone_map.get(pid) == "iPhone"
            else 0
        )

        # split into segments, then resample
        raw_segments = split_into_continuous_segments(acc_pid)

        segments = []
        for seg in raw_segments:
            uniform = resample_segment_to_uniform(seg, SAMPLE_RATE)
            if uniform is not None:
                segments.append(uniform)

        total_uniform_samples = sum(len(s) for s in segments)
        print(
            f"Continuous segments after resample: {len(segments)} "
            f"(uniform samples: {total_uniform_samples:,} @ {SAMPLE_RATE} Hz)"
        )

        # remove each subject's per-axis mean so model doesn't see
        # orientation as a subject-identity cue.
        if len(segments) > 0:
            all_xyz = np.concatenate(
                [s[["x", "y", "z"]].values for s in segments],
                axis=0,
            )
            subj_mean = all_xyz.mean(axis=0)

            for s in segments:
                for i, axis in enumerate(["x", "y", "z"]):
                    s[axis] = s[axis].values - subj_mean[i]


        # tabular
        for segment in segments:

            windows, starts, mids, ends = create_windows(
                segment,
                TAB_STRIDE
            )

            labels, tac_interp = label_windows(
                mids,
                tac_df,
                max_gap_minutes=MAX_TAC_GAP_MINUTES
            )

            valid = labels != -1

            windows = [
                w for w, keep in zip(windows, valid)
                if keep
            ]

            labels = labels[valid]

            for window, label in zip(windows, labels):

                feats = extract_features(window, phone_feature)

                tab_X.append(feats)
                tab_y.append(label)
                tab_pids.append(pid)

        # sequential
        for segment in segments:

            windows, starts, mids, ends = create_windows(
                segment,
                SEQ_STRIDE
            )

            labels, tac_interp = label_windows(
                mids,
                tac_df,
                max_gap_minutes=MAX_TAC_GAP_MINUTES
            )

            valid = labels != -1

            windows = [
                w for w, keep in zip(windows, valid)
                if keep
            ]

            labels = labels[valid]

            for window, label in zip(windows, labels):

                downsampled = resample_poly(
                    window,
                    up=1,
                    down=2,
                    axis=0
                )

                seq_X.append(downsampled.astype(np.float32))
                seq_y.append(label)
                seq_pids.append(pid)

        n_pid_tab = sum(1 for p in tab_pids if p == pid)
        n_pid_drunk = sum(
            1 for p, lbl in zip(tab_pids, tab_y)
            if p == pid and lbl == 1
        )

        print(
            f"Tab windows for {pid}: {n_pid_tab} "
            f"(drunk={n_pid_drunk}, sober={n_pid_tab - n_pid_drunk})"
        )

    # save
    np.save(os.path.join(TAB_DIR, "X.npy"), np.array(tab_X, dtype=np.float32))
    np.save(os.path.join(TAB_DIR, "y.npy"), np.array(tab_y))
    np.save(os.path.join(TAB_DIR, "pids.npy"), np.array(tab_pids))

    np.save(os.path.join(SEQ_DIR, "X.npy"), np.array(seq_X, dtype=np.float32))
    np.save(os.path.join(SEQ_DIR, "y.npy"), np.array(seq_y))
    np.save(os.path.join(SEQ_DIR, "pids.npy"), np.array(seq_pids))

    # summary
    y = np.array(tab_y)

    print("DONE")

    print(f"Total tab windows : {len(y)}")

    drunk = int(np.sum(y == 1))
    sober = int(np.sum(y == 0))

    print(f"Drunk windows : {drunk}")
    print(f"Sober windows : {sober}")

    if len(y) > 0:
        print(f"Drunk % : {100 * np.mean(y):.2f}%")

    print(f"Sequential shape : {np.array(seq_X).shape}")


# raw plots
def plot_raw_signals(out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(BASE_DIR, "project", "plots")
    os.makedirs(out_dir, exist_ok=True)

    acc = load_accelerometer_data()
    pids = load_pids()

    for pid in pids:

        acc_pid = acc[acc["pid"] == pid].sort_values("time")
        if len(acc_pid) == 0:
            continue

        try:
            tac_df = load_tac(pid)
        except FileNotFoundError:
            tac_df = None

        # estimated raw rate (for the title)
        diffs = acc_pid["time"].diff().dt.total_seconds()
        median_dt = diffs.median()
        est_hz = 1.0 / median_dt if median_dt and median_dt > 0 else float("nan")

        # split into segments so plot doesnt draw across gaps
        segments = split_into_continuous_segments(acc_pid)

        fig, ax = plt.subplots(figsize=(13, 4.2))

        for seg in segments:
            ax.plot(seg["time"], seg["x"],
                    color="tab:blue",   lw=0.4, alpha=0.7)
            ax.plot(seg["time"], seg["y"],
                    color="tab:orange", lw=0.4, alpha=0.7)
            ax.plot(seg["time"], seg["z"],
                    color="tab:green",  lw=0.4, alpha=0.7)

        # legend
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], color="tab:blue",   label="x"),
                Line2D([0], [0], color="tab:orange", label="y"),
                Line2D([0], [0], color="tab:green",  label="z"),
            ],
            loc="upper left",
            fontsize=8,
        )

        ax.set_ylabel("acceleration (g)")
        ax.set_ylim(-MAX_ABS_ACCEL, MAX_ABS_ACCEL)
        ax.grid(True, alpha=0.3)

        # TAC overlay
        if tac_df is not None and len(tac_df) > 0:

            ax2 = ax.twinx()

            t_lo = acc_pid["time"].min() - pd.Timedelta(minutes=30)
            t_hi = acc_pid["time"].max() + pd.Timedelta(minutes=30)

            tac_view = tac_df[
                (tac_df["timestamp"] >= t_lo) &
                (tac_df["timestamp"] <= t_hi)
            ]

            ax2.plot(
                tac_view["timestamp"],
                tac_view["TAC_Reading"],
                color="black", lw=1.6,
                label="TAC",
            )
            ax2.axhline(
                0.08,
                color="red", ls="--", lw=1.0, alpha=0.7,
            )

            # shade intoxicated periods (TAC >= 0.08)
            if len(tac_view) > 1:
                tac_t = tac_view["timestamp"].values
                tac_v = tac_view["TAC_Reading"].values
                drunk_mask = tac_v >= 0.08
                # find contiguous drunk regions
                in_drunk = False
                start = None
                for ti, t in enumerate(tac_t):
                    if drunk_mask[ti] and not in_drunk:
                        start = t
                        in_drunk = True
                    elif not drunk_mask[ti] and in_drunk:
                        ax.axvspan(start, t, color="red", alpha=0.08)
                        in_drunk = False
                if in_drunk:
                    ax.axvspan(start, tac_t[-1], color="red", alpha=0.08)

            ax2.set_ylabel("TAC")
            tac_max = max(0.25, float(tac_view["TAC_Reading"].max()) * 1.1)
            ax2.set_ylim(-0.05, tac_max)
            ax2.legend(loc="upper right", fontsize=8)

        # constrain x-axis to the actual recording window
        ax.set_xlim(acc_pid["time"].min(), acc_pid["time"].max())

        n_drunk = 0
        if tac_df is not None:
            n_drunk = int((tac_df["TAC_Reading"] >= 0.08).sum())

        ax.set_title(
            f"{pid}  --  raw signal  "
            f"(raw ~{est_hz:.1f} Hz)",
            fontsize=10,
        )

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{pid}_raw.png")
        plt.savefig(out_path, dpi=110)
        plt.close(fig)

        print(f"saved {out_path}")

if __name__ == "__main__":

    build_and_save()
    plot_raw_signals()