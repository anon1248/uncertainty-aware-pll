""" Module for loading data. """

from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.io import arff, loadmat
from sklearn.model_selection import KFold

# Some more in arff format
UCI_DATA_ARFF = list(sorted(
    glob("data/ucipp/uci/*.arff")
))
UCI_DATA_LABELS_ARFF = [
    path.split("/")[-1].split(".")[0] for path in UCI_DATA_ARFF
]
UCI_LABEL_TO_PATH_ARFF = dict(zip(UCI_DATA_LABELS_ARFF, UCI_DATA_ARFF))

# Selected arff-datasets
SELECTED_ARFF_DATASETS = {
    "artificial-characters",
    "ecoli",
    "first-order-theorem",
    "flare",
    "kr-vs-k",
    "mfeat-fourier",
    "pendigits",
    "semeion",
    "statlog-landsat-satellite",
}

# All real-world datasets
REAL_WORLD_DATA = list(sorted(
    glob("data/realworld-datasets/*.mat")
))
REAL_WORLD_DATA_LABELS = [
    path.split("/")[-1].split(".")[0] for path in REAL_WORLD_DATA
]
REAL_WORLD_LABEL_TO_PATH = dict(zip(REAL_WORLD_DATA_LABELS, REAL_WORLD_DATA))


class Dataset:
    """ A dataset. """

    def __init__(
        self, x_full: np.ndarray, y_full: np.ndarray, y_true: np.ndarray,
        n_samples: int, m_features: int, l_classes: int,
    ) -> None:
        self.x_full = x_full
        self.y_full = y_full
        self.y_true = y_true
        self.n_samples = int(n_samples)
        self.m_features = int(m_features)
        self.l_classes = int(l_classes)

    def augment_targets(
        self,
        rng: np.random.Generator,
        r_candidates: int,
        percent_partially_labeled: float,
        eps_cooccurrence: float,
    ) -> "Dataset":
        """ Augments a supervised dataset with random label candidates.

        Args:
            rng (np.random.Generator): The random number generator.
            r_candidates (int): The candidates.
            percent_partially_labeled (float): The number of instances that is partially labeled.
            eps_cooccurrence (float): The probability of co-occurrence.

        Returns:
            Dataset: The data set.
        """

        # Create co-occurrence pairs
        class_perm = list(map(int, rng.permutation(self.l_classes)))
        class_perm.append(-1)  # Sentinel if odd number of elements
        class_pairs = list(zip(class_perm[::2], class_perm[1::2]))
        co_occ_classes = {}
        for elem1, elem2 in class_pairs:
            co_occ_classes[elem1] = elem2
            co_occ_classes[elem2] = elem1

        # Determine probabilities of item selection
        eps_cooccurrence = max(eps_cooccurrence, 1 / (self.l_classes - 1))
        other_prob = (1 - eps_cooccurrence) / (self.l_classes - 2)

        # Iterate train set and add false-positve labels
        y_full_copy = self.y_full.copy()
        for i in range(self.n_samples):
            # Partially label a percentage of all instances
            if rng.random() < percent_partially_labeled:
                # Compute probabilities for each label
                true_label = int(self.y_true[i])
                co_occ_class = co_occ_classes[true_label]
                if co_occ_class != -1:
                    probs = other_prob * np.ones(self.l_classes)
                    probs[true_label] = 0
                    probs[co_occ_class] = eps_cooccurrence
                else:
                    probs = (1 / (self.l_classes - 1)) * \
                        np.ones(self.l_classes)
                    probs[true_label] = 0

                # Check that probabilities sum to one
                if np.abs(np.sum(probs) - 1) > 1e-10:
                    raise ValueError("Probabilities must sum to one.")

                # Draw candidates
                candidates = list(map(int, rng.choice(
                    self.l_classes, replace=False, p=probs, size=r_candidates,
                )))
                y_full_copy[i, candidates] = 1

        return Dataset(
            self.x_full, y_full_copy, self.y_true,
            self.n_samples, self.m_features, self.l_classes
        )

    def copy(self) -> "Dataset":
        """ Copies the dataset.

        Returns:
            Dataset: The copy.
        """

        return Dataset(
            self.x_full.copy(), self.y_full.copy(), self.y_true.copy(),
            self.n_samples, self.m_features, self.l_classes,
        )


class Datasplit:
    """ A data split. """

    def __init__(
        self, x_train: np.ndarray, x_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        y_true_train: np.ndarray, y_true_test: np.ndarray,
        orig_dataset: Dataset,
        normalize: str = "minmax",
    ) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_true_train = y_true_train
        self.y_true_test = y_true_test
        self.orig_dataset = orig_dataset
        self.normalize = normalize

        if self.normalize == "minmax":
            self.x_train_min = np.min(self.x_train, axis=0)
            self.x_train_max = np.max(self.x_train, axis=0)
            self.x_train_min = np.where(
                self.x_train_min == self.x_train_max, 0, self.x_train_min)
            self.x_train_max = np.where(
                self.x_train_min == self.x_train_max, 1, self.x_train_max)
        elif self.normalize == "normal":
            self.x_train_mean = np.mean(self.x_train, axis=0)
            self.x_train_std = np.std(self.x_train, axis=0)
            self.x_train_std = np.where(
                self.x_train_std == 0, 1, self.x_train_std)

        self.x_train = self._transform(self.x_train)
        self.x_test = self._transform(self.x_test)

    def _transform(self, x_data: np.ndarray) -> np.ndarray:
        """ Normalizes the given data.

        Args:
            x_data (np.ndarray): The data.

        Returns:
            np.ndarray: The normalized data.
        """

        if x_data.shape[0] == 0:
            return x_data

        if self.normalize == "minmax":
            return (
                (x_data - self.x_train_min) /
                (self.x_train_max - self.x_train_min)
            )
        if self.normalize == "normal":
            return (
                (x_data - self.x_train_mean) /
                self.x_train_std
            )
        return x_data

    @classmethod
    def create_random_split_from_dataset(
        cls, dataset: Dataset, split_idx: int,
        rng: np.random.Generator, test_size: float = 0.5,
    ) -> "Datasplit":
        """ Create random split from dataset.

        Args:
            dataset (Dataset): The dataset.
            split_idx (int): The index of the k-fold cross-validation split.
            test_size (float): The test size. Defaults to 0.5.

        Returns:
            Datasplit: The data split.
        """

        # If test size positive, create train-test split
        if test_size > 0:
            k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
            train_indices, test_indices = list(
                k_fold.split(dataset.x_full))[split_idx]
            return Datasplit(
                dataset.x_full[train_indices].copy(),
                dataset.x_full[test_indices].copy(),
                dataset.y_full[train_indices].copy(),
                dataset.y_full[test_indices].copy(),
                dataset.y_true[train_indices].copy(),
                dataset.y_true[test_indices].copy(),
                dataset,
            )

        # Else, just permute instances
        perm = rng.permutation(dataset.x_full.shape[0])
        return Datasplit(
            dataset.x_full[perm].copy(), np.array([]),
            dataset.y_full[perm].copy(), np.array([]),
            dataset.y_true[perm].copy(), np.array([]),
            dataset,
        )

    def copy(self) -> "Datasplit":
        """ Copies the datasplit.

        Returns:
            Datasplit: The copy.
        """

        return Datasplit(
            self.x_train.copy(), self.x_test.copy(), self.y_train.copy(),
            self.y_test.copy(), self.y_true_train.copy(),
            self.y_true_test.copy(), self.orig_dataset.copy(), self.normalize,
        )


def get_all_datasets_arff() -> Dict[str, Dataset]:
    """ Retrieves all UCI datasets.

    Returns:
        Dict[str, Dataset]: Maps names to datasets.
    """

    all_datasets = {}
    for name, path in UCI_LABEL_TO_PATH_ARFF.items():
        if name not in SELECTED_ARFF_DATASETS:
            continue

        # Load dataset
        data, meta = arff.loadarff(path)
        dataframe = pd.DataFrame.from_records(data)
        l_classes = dataframe["Class"].unique().shape[0]

        for col, col_type in zip(
            map(str, meta.names()),
            map(str, meta.types())
        ):
            if col == "Class":
                dataframe["Class"] = pd.Categorical(
                    dataframe["Class"]).codes.astype(int)
            elif col_type == "nominal":
                # Onehot encode column
                onehot_df = dataframe[col].astype(str).str.get_dummies()
                for i, onehot_col in enumerate(map(str, onehot_df)):
                    dataframe[f"{col}_{i}"] = onehot_df[
                        onehot_col].astype(float)
                    dataframe = dataframe.copy()
                dataframe.drop(col, axis=1, inplace=True)
            elif col_type == "numeric":
                # Parse as float
                cols = list(dataframe.columns)
                cols.remove(col)
                cols.append(col)
                dataframe[col] = dataframe[col].astype(float)
                dataframe = dataframe[cols]
            else:
                # Unknown col_type
                raise ValueError(f"Unknown column type: {col_type}")

        # Extract values
        x_raw = dataframe.loc[:, dataframe.columns != "Class"].values
        y_raw = dataframe["Class"].values

        # Exclude zero-variance features
        x_raw = x_raw[:, x_raw.var(axis=0) > 1e-30]

        # Extract cardinalities
        n_samples = x_raw.shape[0]
        m_features = x_raw.shape[1]
        l_classes = np.unique(y_raw).shape[0]

        # Partial label vector
        pl_vec = np.zeros((n_samples, l_classes), dtype=int)
        for i, y_val in enumerate(y_raw):
            pl_vec[i, y_val] = 1

        # Store dataset
        all_datasets[name] = Dataset(
            x_raw, pl_vec, y_raw, n_samples, m_features, l_classes,
        )

    return all_datasets


def get_all_rl_datasets() -> Dict[str, Dataset]:
    """ Retrieves all real-world datasets.

    Returns:
        Dict[str, Dataset]: The real-world datasets.
    """

    # Coerce data into dense array
    def coerce(data) -> np.ndarray:
        try:
            return data.toarray()
        except:  # pylint: disable=bare-except
            return data

    # Extract datasets
    all_datasets = {}
    for name, path in sorted(REAL_WORLD_LABEL_TO_PATH.items()):
        # Extract raw data
        raw_mat_data = loadmat(path)
        x_raw = coerce(raw_mat_data["data"])
        y_partial_raw = coerce(raw_mat_data["partial_target"].transpose())
        y_true_raw = np.argmax(
            coerce(raw_mat_data["target"].transpose()), axis=1)

        # Number of classes representing 99% of all occurrences
        num_classes = int(np.where(np.cumsum(
            np.array(list(reversed(list(np.sort(np.count_nonzero(
                coerce(raw_mat_data["target"].transpose()), axis=0
            )))))) / y_true_raw.shape[0]) > 0.99
        )[0].min())
        num_classes = min(num_classes + 1, int(y_partial_raw.shape[1]))
        classes_in_use = set(map(int, np.sort(np.argsort(
            np.count_nonzero(y_partial_raw, axis=0))[-num_classes:])))

        # Collect all relevant data
        x_list = []
        y_partial_list = []
        y_true_list: List[int] = []
        mask = np.array(list(sorted(list(classes_in_use))))
        for x_row, y_partial_row, y_true_row in zip(
            x_raw, y_partial_raw, y_true_raw,
        ):
            if int(y_true_row) in classes_in_use:
                x_list.append(x_row)
                y_partial_list.append(y_partial_row[mask])
                y_true_list.append(int(np.where(
                    mask == int(y_true_row))[0][0]))
        x_arr = np.array(x_list)
        y_partial_arr = np.array(y_partial_list)
        y_true_arr = np.array(y_true_list)
        x_arr = x_arr[:, x_arr.var(axis=0) > 1e-30].copy()

        # Store dataset
        all_datasets[name] = Dataset(
            x_arr, y_partial_arr, y_true_arr,
            x_arr.shape[0], x_arr.shape[1], y_partial_arr.shape[1],
        )

    return all_datasets
