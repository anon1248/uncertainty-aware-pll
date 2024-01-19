""" Generates the plots in Section 6.2. """

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

datasets_used = pd.read_csv("results/datasets_used.csv", index_col="idx")
results_df = pd.read_csv("results/results.csv")
results_df = results_df[results_df["dataset"].isin(
    datasets_used["dataset"].unique())]

results_grouped = results_df.groupby([
    "dataset", "algo", "train_frac", "r_candidates",
    "p_partially_labeled", "eps_coocc",
]).aggregate(["mean", "std"])
results_grouped.columns = ["_".join(col) for col in results_grouped.columns]
results_grouped.drop(["seed_mean", "seed_std"], axis=1, inplace=True)
results_grouped = results_grouped.reset_index()

results_per_dataset = results_df.groupby([
    "dataset", "algo", "train_frac",
]).aggregate(["mean", "std"])
results_per_dataset.columns = ["_".join(col)
                               for col in results_per_dataset.columns]
results_per_dataset.drop([
    "seed_mean", "seed_std", "r_candidates_mean", "r_candidates_std",
    "p_partially_labeled_mean", "p_partially_labeled_std", "eps_coocc_mean", "eps_coocc_std",
], axis=1, inplace=True)
results_per_dataset = results_per_dataset.reset_index()

random.seed(42)
plt.style.use("tableau-colorblind10")

fs = 9
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("savefig", bbox="tight")

plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)

fig, axes = plt.subplots(1, 1, figsize=(2.5, 2))

fmts = [
    "o", "v", "^", "<", ">", "s", "P", "h", "D",
]
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

ds_order = {
    "artificial-characters": 0,
    "kr-vs-k": 1,
    "ecoli": 2,
    "first-order-theorem": 3,
    "mfeat-fourier": 4,
    "pendigits": 5,
    "semeion": 6,
    "statlog-landsat-satellite": 7,
    "flare": 8,
}

algos_list = sorted(
    list(results_per_dataset["algo"].unique()), key=lambda s: s[-4:])

axes.grid(alpha=0.2)
plot_labels = []
for i, (key, fmt) in enumerate(zip(algos_list, fmts)):
    if key not in to_display:
        continue
    x = np.array(list(map(ds_order.get, map(str, list(results_per_dataset.loc[
        (results_per_dataset["algo"] == key),
        "dataset"
    ]))))) - 0.2 + (i / (len(algos_list) - 1)) * 0.4
    y = results_per_dataset.loc[
        (results_per_dataset["algo"] == key),
        "test_mcc_mean"
    ].values
    y_std = results_per_dataset.loc[
        (results_per_dataset["algo"] == key),
        "test_mcc_std"
    ].values
    axes.errorbar(x, y, yerr=y_std, fmt=fmt, markersize=4, zorder=(
        10 if key == "dst-pll" else (9 if key == "pl-knn-2005" else 2)))
    plot_labels.append(to_display[key])

axes.set_ylim(0.0, 1.0)
# axes[1].set_aspect(8)
axes.set_xticks(list(range(9)), list(range(1, 10)))

axes.set_xlabel("Dataset", fontsize=fs)
axes.set_ylabel("Test-Set MCC", fontsize=fs)

# axes.legend(plot_labels, loc="lower left", bbox_to_anchor=(1.06, -0.04), fontsize=fs-1)

axes.set_position([0.1, 0.1, 0.62, 0.71])

plt.savefig("paper/plots/perf-mcc.pdf", transparent=True)

# -------------------

random.seed(42)
plt.style.use("tableau-colorblind10")

fs = 9
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("savefig", bbox="tight")

plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)

fig, axes = plt.subplots(1, 1, figsize=(2.5, 2))

fmts = [
    "o", "v", "^", "<", ">", "s", "P", "h", "D",
]
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

ds_order = {
    "artificial-characters": 0,
    "kr-vs-k": 1,
    "ecoli": 2,
    "first-order-theorem": 3,
    "mfeat-fourier": 4,
    "pendigits": 5,
    "semeion": 6,
    "statlog-landsat-satellite": 7,
    "flare": 8,
}

algos_list = sorted(
    list(results_per_dataset["algo"].unique()), key=lambda s: s[-4:])

# --------------------------

axes.grid(alpha=0.2)
plot_labels = []
for i, (key, fmt) in enumerate(zip(algos_list, fmts)):
    if key not in to_display:
        continue
    x = np.array(list(map(ds_order.get, map(str, list(results_per_dataset.loc[
        (results_per_dataset["algo"] == key),
        "dataset"
    ]))))) - 0.2 + (i / (len(algos_list) - 1)) * 0.4
    y = results_per_dataset.loc[
        (results_per_dataset["algo"] == key),
        "test_acc_mean"
    ].values
    y_std = results_per_dataset.loc[
        (results_per_dataset["algo"] == key),
        "test_acc_std"
    ].values
    axes.errorbar(x, y, yerr=y_std, fmt=fmt, markersize=4, zorder=(
        10 if key == "dst-pll" else (9 if key == "pl-knn-2005" else 2)))
    plot_labels.append(to_display[key])

axes.set_ylim(0.0, 1.0)
axes.set_xticks(list(range(9)), list(range(1, 10)))

axes.set_xlabel("Dataset", fontsize=fs)
axes.set_ylabel("Test-Set Accuracy", fontsize=fs)

# axes.legend(plot_labels, loc="lower left", bbox_to_anchor=(1.06, -0.04), fontsize=fs-1)

axes.set_position([0.1, 0.1, 0.62, 0.71])

plt.savefig("paper/plots/perf-acc.pdf", transparent=True)

# -------------------------

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

fig, axes = plt.subplots(1, 1, figsize=(2.5, 2))

betas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

to_display = {
    # "chance": "\\textsc{Chance",
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

offset_to_disp_test = {
    # "chance": (0.02, 0.02),
    "dst-pll": (0.04, -0.027),
    "ipal-2015": (-0.2, -0.026),
    "sure-2019": (0.04, -0.027),
    "pl-knn-2005": (-0.105, -0.095),
    "lsb-cmm-2012": (-0.39, -0.026),
    "m3pl-2016": (0.04, -0.033),
    "pl-svm-2008": (0.04, -0.027),
    "clpl-2011": (-0.22, -0.027),
    "paloc-2018": (0.04, -0.027),
    # "perfect": (-0.25, -0.08),
}

fmts = [
    "o", "v", "^", "<", ">", "s", "P", "h", "D",
]
categories = {
    # "chance": 5,
    "dst-pll": 0,
    "ipal-2015": 5,
    "sure-2019": 8,
    "pl-knn-2005": 1,
    "lsb-cmm-2012": 4,
    "m3pl-2016": 6,
    "pl-svm-2008": 2,
    "clpl-2011": 3,
    "paloc-2018": 7,
    # "perfect": 5,
}

# print("MCC")
# print(" " * 23 + " ".join([f"{beta:.3f}" for beta in betas]))

axes.grid(alpha=0.2)
for i, (idx, row) in enumerate(agg_df.iterrows()):
    if str(idx) not in to_display:
        continue
    x = float(row["test_%sure"])
    x_std = float(row["test_%sure_std"])
    y = float(row["test_mcc_sure"])
    y_std = float(row["test_mcc_sure_std"])
    t = to_display[str(idx)]
    off_x, off_y = offset_to_disp_test[str(idx)]
    cat_idx = categories[str(idx)]
    axes.errorbar(x, y, xerr=x_std, yerr=y_std, fmt=fmts[cat_idx], color=f"C{cat_idx}", markersize=4, zorder=(
        10 if key == "dst-pll" else (9 if key == "pl-knn-2005" else 2)))
    axes.text(x + off_x, y + off_y, t, fontsize=fs)
    # res = sorted([
    #     f"{(1 - beta) * x + beta * y:.3f}"
    #     for beta in betas
    # ])
    # print(f"{str(idx): >20} - {' '.join(res)}")

# axes[1].plot(1, 1, fmts[3], color="C5")
# off_x, off_y = offset_to_disp_test["perfect"]
# axes[1].text(1 + off_x, 1 + off_y, "\\textsc{Perfect}", fontsize=fs)
axes.set_xlim(0, 1)
axes.set_ylim(0, 1)
axes.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes.set_xlabel("Frac. of confident predictions", fontsize=fs)
axes.set_ylabel("MCC of confident pred.", fontsize=fs)

axes.set_position([0.1, 0.1, 0.62, 0.71])

# print()
# print(axes[0].get_position())
# print(axes[1].get_position())

plt.savefig("paper/plots/confident-predictions-mcc.pdf", transparent=True)

# ---------------------

plt.style.use("tableau-colorblind10")

fs = 9
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("savefig", bbox="tight")

plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)

fig, axes = plt.subplots(1, 1, figsize=(2.5, 2))

betas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

to_display = {
    # "chance": "\\textsc{Chance",
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

fmts = [
    "o", "v", "^", "<", ">", "s", "P", "h", "D",
]
categories = {
    # "chance": 5,
    "dst-pll": 0,
    "ipal-2015": 5,
    "sure-2019": 8,
    "pl-knn-2005": 1,
    "lsb-cmm-2012": 4,
    "m3pl-2016": 6,
    "pl-svm-2008": 2,
    "clpl-2011": 3,
    "paloc-2018": 7,
    # "perfect": 5,
}

offset_to_disp_test = {
    # "chance": (0.02, 0.02),
    "dst-pll": (0.04, -0.05),
    "ipal-2015": (-0.09, -0.085),
    "sure-2019": (0.04, -0.027),
    "pl-knn-2005": (-0.105, -0.09),
    "lsb-cmm-2012": (0.006, 0.015),
    "m3pl-2016": (-0.26, -0.02),
    "pl-svm-2008": (-0.1075, -0.11),
    "clpl-2011": (-0.09, -0.11),
    "paloc-2018": (0.04, -0.027),
    # "perfect": (-0.25, -0.07),
}

# print("Acc")
# print(" " * 23 + " ".join([f"{beta:.3f}" for beta in betas]))

axes.grid(alpha=0.2)
plot_labels = []
for idx, row in agg_df.iterrows():
    if str(idx) not in to_display:
        continue
    x = float(row["test_%sure"])
    x_std = float(row["test_%sure_std"])
    y = float(row["test_acc_sure"])
    y_std = float(row["test_acc_sure_std"])
    t = to_display[str(idx)]
    off_x, off_y = offset_to_disp_test[str(idx)]
    cat_idx = categories[str(idx)]
    axes.errorbar(
        x, y, xerr=x_std, yerr=y_std, fmt=fmts[cat_idx],
        color=f"C{cat_idx}", markersize=4, zorder=(
            10 if key == "dst-pll" else (9 if key == "pl-knn-2005" else 2)))
    axes.text(x + off_x, y + off_y, t, fontsize=fs)
    plot_labels.append(t)
    # res = sorted([
    #     f"{(1 - beta) * x + beta * y:.3f}"
    #     for beta in betas
    # ])
    # print(f"{str(idx): >20} - {' '.join(res)}")
# axes[0].plot(1, 1, fmts[3], color="C5")
# off_x, off_y = offset_to_disp_test["perfect"]
# axes[0].text(1 + off_x, 1 + off_y, "\\textsc{Perfect}", fontsize=fs)
axes.set_xlim(0, 1)
axes.set_ylim(0, 1)
axes.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes.set_xlabel("Frac. of confident predictions", fontsize=fs)
axes.set_ylabel("Acc. of confident pred.", fontsize=fs)

# legend_elements = [
#     Line2D([0], [0], marker=fmts[0], markersize=7, color="w", label="Built-in Uncertainty", markerfacecolor="C0"),
#     Line2D([0], [0], marker=fmts[1], markersize=7, color="w", label="Scoring-based", markerfacecolor="C1"),
#     Line2D([0], [0], marker=fmts[3], markersize=7, color="w", label="Distance-based", markerfacecolor="C3"),
#     #Line2D([0], [0], marker=fmts[5], markersize=7, color="w", label="Baseline", markerfacecolor="C5"),
# ]
# axes[1].legend(
#     handles=legend_elements, loc="lower left",
#     bbox_to_anchor=(1.06, -0.04), fontsize=fs-1, ncol=1,
#     handletextpad=0.2, columnspacing=0.1,
# )

axes.legend(plot_labels, loc="lower left",
            bbox_to_anchor=(1.06, -0.04), fontsize=fs-1)

axes.set_position([0.1, 0.1, 0.62, 0.71])

# print()
# print(axes[0].get_position())
# print(axes[1].get_position())

plt.savefig("paper/plots/confident-predictions-acc.pdf", transparent=True)

# ---------------------

results_rt_grouped = pd.merge(
    results_df.loc[:, ["dataset", "algo", "runtime"]],
    datasets_used.loc[:, ["dataset", "n_samples"]],
    on="dataset",
).groupby(["dataset", "algo"]).aggregate(["mean", "std"])
results_rt_grouped.columns = ["_".join(col)
                              for col in results_rt_grouped.columns]
results_rt_grouped.drop(["n_samples_std"], axis=1, inplace=True)
results_rt_grouped = results_rt_grouped.sort_values(
    by="n_samples_mean").reset_index()


def get_rt_for_algo(algo: str):
    data = [
        (n, rt, rt_std)
        for _, (n, rt, rt_std) in results_rt_grouped.loc[
            results_rt_grouped["algo"] == algo,
            ["n_samples_mean", "runtime_mean", "runtime_std"],
        ].iterrows()
    ]
    return [x for x, _, _ in data], [y for _, y, _ in data], [z for _, _, z in data]


plt.style.use("tableau-colorblind10")

fs = 9
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("savefig", bbox="tight")

plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)

fig, axes = plt.subplots(1, 1, figsize=(2.5, 2))

fmts = [
    "-o", "-v", "-^", "-<", "->", "-s", "-P", "-h", "-D",
]
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

algos_list = sorted(
    list(results_rt_grouped["algo"].unique()), key=lambda s: s[-4:])

axes.grid(alpha=0.2)
plot_labels = []
for key, fmt in zip(algos_list, fmts):
    if key not in to_display:
        continue
    x, y, y_err = get_rt_for_algo(key)
    axes.errorbar(x, y, yerr=y_std, fmt=fmt, markersize=4, linewidth=1.25)
    plot_labels.append(to_display[key])

axes.set_xscale("log")
axes.set_yscale("log")
# axes.set_aspect("equal")
axes.set_xlim(300, 31000)
axes.set_ylim(0.001, 100000)
# axes.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes.set_yticks([0.001, 0.1, 10, 1000, 100000])
# axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

axes.set_xlabel("Number of examples", fontsize=fs)
axes.set_ylabel("Runtime (s)", fontsize=fs)
leg = axes.legend(plot_labels, loc="lower left", bbox_to_anchor=(1.01, -0.045),
                  fontsize=fs-1)

axes.set_position([0.1, 0.1, 0.62, 0.71])

# plt.tight_layout()
plt.savefig("paper/plots/runtime.pdf", transparent=True)
