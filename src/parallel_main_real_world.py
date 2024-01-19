""" Main module. """

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from partial_label_learning.data import get_all_rl_datasets
from partial_label_learning.experiments import (COLUMNS, print_datasets,
                                                run_experiment)


def main() -> None:
    """ Main method. """

    # Print datasets
    datasets = get_all_rl_datasets()
    print_datasets(datasets, prefix="real_world_")

    # Run experiments in parallel
    resres = Parallel(n_jobs=-1)(
        delayed(run_experiment)(
            dataset_name, dataset, train_frac,
            -1, -1, -1, seed, False, False, "all",
        )
        for dataset_name, dataset in tqdm(datasets.items())
        for train_frac in [0.8]
        for seed in range(5)
    )

    # Extract results
    res = sorted([row for resrow in resres for row in resrow])
    pd.DataFrame.from_records(res, columns=COLUMNS).to_csv(
        "results/results_real_world.csv", index=False)


if __name__ == "__main__":
    main()
