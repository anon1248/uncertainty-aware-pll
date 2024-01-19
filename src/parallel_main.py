""" Main module. """

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from partial_label_learning.data import get_all_datasets_arff
from partial_label_learning.experiments import (COLUMNS, print_datasets,
                                                run_experiment)


def main() -> None:
    """ Main method. """

    datasets = get_all_datasets_arff()
    print_datasets(datasets)

    # Run experiments in parallel
    resres = Parallel(n_jobs=-1)(
        delayed(run_experiment)(
            dataset_name, dataset, train_frac,
            r_candidates, percent_partially_labeled,
            eps_coocc, seed, False, True, "all",
        )
        for dataset_name, dataset in tqdm(datasets.items())
        for train_frac in [0.8]
        for r_candidates in [1, 2, 3]
        for percent_partially_labeled in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for eps_coocc in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        if eps_coocc == 0.0 or (eps_coocc > 0.0 and r_candidates == 1)
        for seed in range(5)
    )

    # Extract results
    res = sorted([row for resrow in resres for row in resrow])
    pd.DataFrame.from_records(res, columns=COLUMNS).to_csv(
        "results/results.csv", index=False)


if __name__ == "__main__":
    main()
