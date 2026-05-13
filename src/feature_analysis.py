import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import RESULTS_DIR

X = np.load("project/processed/tab/X.npy")
y = np.load("project/processed/tab/y.npy")

stats = ["mean", "std", "min", "max", "range", "zcr", "domfreq"]

feature_names = []
for axis in ["x", "y", "z"]:

    for stat in stats:

        name = axis + "_" + stat
        feature_names.append(name)

feature_names.append("phone")


# remove phone feature
new_feature_names = []
keep_columns = []

for name in feature_names:

    if name != "phone":
        new_feature_names.append(name)
        keep_columns.append(True)

    else:
        keep_columns.append(False)

feature_names = new_feature_names

# keep only desired columns in X
X = X[:, keep_columns]

num_windows = len(y)
percent_drunk = y.mean() * 100

print("Loaded", num_windows, "windows")
print("Percent drunk:", round(percent_drunk, 1))


# class averages
X_sober = X[y == 0]
X_drunk = X[y == 1]

# avg feature values
mean_sober = X_sober.mean(axis=0)
mean_drunk = X_drunk.mean(axis=0)

# std across all data
std_all = X.std(axis=0)

for i in range(len(std_all)):

    if std_all[i] == 0:
        std_all[i] = 1.0

# compute feature gaps
# gap tells us how different the feature is
# positive gap:
# feature higher in drunk windows
# negative gap:
# feature higher in sober windows

gap = []

for i in range(len(feature_names)):

    diff = mean_drunk[i] - mean_sober[i]

    scaled_diff = diff / std_all[i]

    gap.append(scaled_diff)

gap = np.array(gap)


# features higher in sober 
# we want the most negative gaps

sober_features = []

for i in range(len(gap)):

    if gap[i] < 0:
        sober_features.append((gap[i], i))

# sort from most negative upward
sober_features.sort()

# top 3
top3 = sober_features[:3]

print("\nTop 3 features where higher value -> sober:\n")

for value, index in top3:

    print(feature_names[index])

# plot

def plot_feature(ax, feature_index):

    # get feature values
    sober_values = X[y == 0, feature_index]
    drunk_values = X[y == 1, feature_index]

    # robust plotting range
    low = np.percentile(X[:, feature_index], 1)
    high = np.percentile(X[:, feature_index], 99)

    bins = np.linspace(low, high, 40)

    # sober histogram
    ax.hist(
        sober_values,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Sober"
    )

    # drunk histogram
    ax.hist(
        drunk_values,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Drunk"
    )

    ax.set_title(feature_names[feature_index])
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density")
    ax.legend()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, pair in zip(axes, top3):

    gap_value, feature_index = pair

    plot_feature(ax, feature_index)

plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, "feature_analysis.png")

plt.savefig(save_path, dpi=150)

plt.close()

print("\nSaved figure to:")
print(save_path)