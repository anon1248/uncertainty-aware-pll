""" Generates the plots in Section 6.3. """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datasets_used = pd.read_csv(
    "results/real_world_datasets_used.csv", index_col="idx")
results_df = pd.read_csv("results/results_real_world.csv")
results_df = results_df[results_df["dataset"].isin(
    datasets_used["dataset"].unique())]

results_grouped = results_df.groupby([
    "dataset", "algo", "train_frac", "r_candidates",
    "p_partially_labeled", "eps_coocc",
]).aggregate(["mean", "std"])
results_grouped.columns = ["_".join(col) for col in results_grouped.columns]
results_grouped.drop(["seed_mean", "seed_std"], axis=1, inplace=True)
results_grouped = results_grouped.reset_index()

agg_df = results_grouped[
    (results_grouped["train_frac"] == 0.8)
].groupby(["algo"])[[
    "train_mcc_mean", "test_mcc_mean", "test_acc_sure_mean", "test_acc_sure_std",
    "train_frac_guessing_mean", "test_frac_guessing_mean",
    "train_frac_sure_mean", "test_frac_sure_mean", "test_frac_sure_std",
    "train_mcc_sure_mean", "test_mcc_sure_mean", "test_mcc_sure_std",
    "runtime_mean"
]].mean().sort_values(by="algo", key=lambda x: x.str[-4:-1]).rename(columns={
    "train_mcc_mean": "train_mcc", "test_mcc_mean": "test_mcc",
    "train_frac_guessing_mean": "train_guess", "test_frac_guessing_mean": "test_guess",
    "train_frac_sure_mean": "train_%sure", "test_frac_sure_mean": "test_%sure",
    "train_mcc_sure_mean": "train_mcc_sure", "test_mcc_sure_mean": "test_mcc_sure",
    "test_frac_sure_std": "test_%sure_std", "test_mcc_sure_std": "test_mcc_sure_std",
    "runtime_mean": "rt", "test_acc_sure_mean": "test_acc_sure",
    "test_acc_sure_std": "test_acc_sure_std",
})

plt.style.use("tableau-colorblind10")

fs = 9
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("savefig", bbox="tight")

plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)

fmts = [
    "o", "v", "^", "<", ">", "s", "P", "h", "D",
]

result_df = results_grouped

datasets_list = [
    "bird-song",
    "mir-flickr",
    "lost",
    "msrc-v2",
    "yahoo-news",
]
algos_list = sorted(list(result_df["algo"].unique()), key=lambda s: s[-4:])

to_display = {
    # "chance": "chance",
    "dst-pll": "\\textsc{Dst-Pll}",
    "ipal-2015": "\\textsc{Ipal}",
    "sure-2019": "\\textsc{Sure}",
    "pl-knn-2005": "\\textsc{Pl-Knn}",
    "lsb-cmm-2012": "\\textsc{Lsb-Cmm}",
    "m3pl-2016": "\\textsc{M3Pl}",
    "pl-svm-2008": "\\textsc{Pl-Svm}",
    "clpl-2011": "\\textsc{Clpl}",
    "paloc-2018": "\\textsc{Paloc}",
}

w = 7.4
fig, axes = plt.subplots(1, 3, figsize=(w, w * 0.4))

# Select data
data = result_df

# Gather data for each boxplot
plot_labels = []
plot_data = []
plot_errors = []
for algo in algos_list:
    if algo == "chance":
        continue

    plot_labels.append(to_display[algo])
    plot_data.append([
        float(data[
            (data["dataset"] == d) &
            (data["algo"] == algo)
        ]["test_mcc_mean"].iloc[0])
        for d in datasets_list
    ])
    plot_errors.append([
        float(data[
            (data["dataset"] == d) &
            (data["algo"] == algo)
        ]["test_mcc_std"].iloc[0])
        for d in datasets_list
    ])

# Plot boxplots
axes[0].grid(alpha=0.3)
for i, (dt, err) in enumerate(zip(plot_data, plot_errors)):
    axes[0].errorbar(np.arange(5) - 0.2 + (i / (len(plot_data) - 1)) * 0.4,
                     np.clip(dt, 0.0, 1.0), yerr=err, fmt=fmts[i],
                     zorder=(10 if i == 0 else 1), markersize=4.5)
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("Test-Set MCC", fontsize=fs)
axes[0].set_xlabel("Dataset", fontsize=fs)
axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [
                   0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs)
axes[0].set_xticks([0, 1, 2, 3, 4], ["bird", "flickr",
                   "lost", "msrc", "yahoo"], fontsize=fs)

# Gather data for each boxplot
plot_labels = []
plot_data = []
plot_errors = []
for algo in algos_list:
    if algo == "chance":
        continue

    plot_labels.append(to_display[algo])
    plot_data.append([
        float(data[
            (data["dataset"] == d) &
            (data["algo"] == algo)
        ]["test_frac_sure_mean"].iloc[0])
        for d in datasets_list
    ])
    plot_errors.append([
        float(data[
            (data["dataset"] == d) &
            (data["algo"] == algo)
        ]["test_frac_sure_std"].iloc[0])
        for d in datasets_list
    ])

# Plot boxplots
axes[1].grid(alpha=0.3)
for i, (dt, err) in enumerate(zip(plot_data, plot_errors)):
    axes[1].errorbar(np.arange(5) - 0.2 + (i / (len(plot_data) - 1)) * 0.4,
                     np.clip(dt, 0.0, 1.0), yerr=err, fmt=fmts[i],
                     zorder=(10 if i == 0 else 1), markersize=4.5)
axes[1].set_ylim(0, 1.0)
axes[1].set_ylabel("Frac.~of confident pred.", fontsize=fs)
axes[1].set_xlabel("Dataset", fontsize=fs)
axes[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [
                   0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs)
axes[1].set_xticks([0, 1, 2, 3, 4], ["bird", "flickr",
                   "lost", "msrc", "yahoo"], fontsize=fs)

# Gather data for each boxplot
plot_labels = []
plot_data = []
plot_errors = []
for algo in algos_list:
    if algo == "chance":
        continue

    plot_labels.append(to_display[algo])
    plot_data.append([
        float(data[
            (data["dataset"] == d) &
            (data["algo"] == algo)
        ]["test_mcc_sure_mean"].iloc[0])
        for d in datasets_list
    ])
    plot_errors.append([
        float(data[
            (data["dataset"] == d) &
            (data["algo"] == algo)
        ]["test_mcc_sure_std"].iloc[0])
        for d in datasets_list
    ])

# Plot boxplots
axes[2].grid(alpha=0.3)
for i, (dt, err) in enumerate(zip(plot_data, plot_errors)):
    axes[2].errorbar(np.arange(5) - 0.2 + (i / (len(plot_data) - 1)) * 0.4,
                     np.clip(dt, 0.0, 1.0), yerr=err, fmt=fmts[i], label=plot_labels[i],
                     zorder=(10 if i == 0 else 1), markersize=4.5)
axes[2].set_ylim(0, 1.0)
axes[2].set_ylabel("MCC of confident pred.", fontsize=fs)
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.99, 0.078),
           fontsize=fs-1, ncol=1)
axes[2].set_xlabel("Dataset", fontsize=fs)
axes[2].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [
                   0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs)
axes[2].set_xticks([0, 1, 2, 3, 4], ["bird", "flickr",
                   "lost", "msrc", "yahoo"], fontsize=fs)

# plt.tight_layout()
aspect = 0.71 / 0.33
axes[0].set_position([0.1, 0.1, 0.22, 0.22 * aspect])
axes[1].set_position([0.4, 0.1, 0.22, 0.22 * aspect])
axes[2].set_position([0.7, 0.1, 0.22, 0.22 * aspect])

plt.savefig("paper/plots/real-world-overview.pdf")

results_sens_df = pd.read_csv("results/results_sensitivity.csv")
results_sens_df = results_sens_df[results_sens_df["dataset"].isin(
    datasets_used["dataset"].unique())]

results_sens_grouped = results_sens_df.groupby([
    "dataset", "algo", "k",
]).aggregate(["mean", "std"])
results_sens_grouped.columns = [
    "_".join(col) for col in results_sens_grouped.columns]
results_sens_grouped.drop(["seed_mean", "seed_std"], axis=1, inplace=True)
results_sens_grouped = results_sens_grouped.reset_index()

plt.style.use("tableau-colorblind10")

fmts = [
    "-", "--", ":", "-.", (5, (10, 3)),
]

datasets_list = {
    (0, "bird-song"): "bird",
    (1, "mir-flickr"): "flickr",
    (2, "lost"): "lost",
    (3, "msrc-v2"): "msrc",
    (4, "yahoo-news"): "yahoo",
}

w = 7.4
fig, axes = plt.subplots(1, 3, figsize=(w, w * 0.363))

# Select data
data = results_sens_grouped

# Gather data for each boxplot
plot_labels = []
plot_data = []
plot_errors = []
for (i, ds), name in sorted(list(datasets_list.items())):
    plot_labels.append(name)
    plot_data.append(data[(data["dataset"] == ds) &
                     (data["k"] <= 50)]["test_mcc_mean"])
    plot_errors.append(data[(data["dataset"] == ds) &
                       (data["k"] <= 50)]["test_mcc_std"])

# Plot boxplots
axes[0].grid(alpha=0.3)
for i, (dt, err) in enumerate(zip(plot_data, plot_errors)):
    axes[0].fill_between(np.arange(1, 51), np.clip(dt - err, 0.0, 1.0),
                         np.clip(dt + err, 0.0, 1.0), color=f"C{i}", alpha=0.1)
    axes[0].plot(np.arange(1, 51), np.clip(dt, 0.0, 1.0),
                 linestyle=fmts[i], color=f"C{i}")
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("Test-Set MCC", fontsize=fs)
axes[0].set_xlabel("Number of Neighbors $k$", fontsize=fs)
axes[0].set_aspect(40)
axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [
                   0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs)
axes[0].set_xlim(0, 50)
axes[0].set_xticks([0, 10, 20, 30, 40, 50], [
                   0, 10, 20, 30, 40, 50], fontsize=fs)

# Gather data for each boxplot
plot_labels = []
plot_data = []
plot_errors = []
for (i, ds), name in sorted(list(datasets_list.items())):
    plot_labels.append(name)
    plot_data.append(data[(data["dataset"] == ds) & (
        data["k"] <= 50)]["test_frac_sure_mean"])
    plot_errors.append(data[(data["dataset"] == ds) & (
        data["k"] <= 50)]["test_frac_sure_std"])

# Plot boxplots
axes[1].grid(alpha=0.3)
for i, (dt, err) in enumerate(zip(plot_data, plot_errors)):
    axes[1].fill_between(np.arange(1, 51), np.clip(dt - err, 0.0, 1.0),
                         np.clip(dt + err, 0.0, 1.0), color=f"C{i}", alpha=0.1)
    axes[1].plot(np.arange(1, 51), np.clip(dt, 0.0, 1.0),
                 linestyle=fmts[i], color=f"C{i}")
axes[1].set_ylim(0, 1.0)
axes[1].set_ylabel("Frac.~of confident pred.", fontsize=fs)
axes[1].set_xlabel("Number of Neighbors $k$", fontsize=fs)
axes[1].set_aspect(40)
axes[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [
                   0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs)
axes[1].set_xlim(0, 50)
axes[1].set_xticks([0, 10, 20, 30, 40, 50], [
                   0, 10, 20, 30, 40, 50], fontsize=fs)

# Gather data for each boxplot
plot_labels = []
plot_data = []
plot_errors = []
for (i, ds), name in sorted(list(datasets_list.items())):
    plot_labels.append(name)
    plot_data.append(data[(data["dataset"] == ds) & (
        data["k"] <= 50)]["test_mcc_sure_mean"])
    plot_errors.append(data[(data["dataset"] == ds) & (
        data["k"] <= 50)]["test_mcc_sure_std"])

# Plot boxplots
axes[2].grid(alpha=0.3)
for i, (dt, err) in enumerate(zip(plot_data, plot_errors)):
    axes[2].fill_between(np.arange(1, 51), np.clip(dt - err, 0.0, 1.0),
                         np.clip(dt + err, 0.0, 1.0), color=f"C{i}", alpha=0.1)
    axes[2].plot(np.arange(1, 51), np.clip(dt, 0.0, 1.0), linestyle=fmts[i],
                 color=f"C{i}", label=plot_labels[i])
axes[2].set_ylim(0, 1.0)
axes[2].set_ylabel("MCC of confident pred.", fontsize=fs)
axes[2].set_xlabel("Number of Neighbors $k$", fontsize=fs)
axes[2].set_aspect(40)
axes[2].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [
                   0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs)
axes[2].set_xlim(0, 50)
axes[2].set_xticks([0, 10, 20, 30, 40, 50], [
                   0, 10, 20, 30, 40, 50], fontsize=fs)

handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.16), fontsize=fs-1, ncol=5)

# plt.tight_layout()
aspect = 0.71 / 0.33
axes[0].set_position([0.1, 0.1, 0.22, 0.22 * aspect])
axes[1].set_position([0.4, 0.1, 0.22, 0.22 * aspect])
axes[2].set_position([0.7, 0.1, 0.22, 0.22 * aspect])

plt.savefig("paper/plots/real-world-sensitivity.pdf")
