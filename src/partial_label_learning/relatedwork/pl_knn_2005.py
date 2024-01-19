""" Module for PL-KNN. """

from typing import List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class PlKnn:
    """
    PL-KNN by Hüllermeier and Beringer,
    "Learning from Ambiguously Labeled Examples."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        n_neighbors: int = 10,
        calibration: float = 1.0,
    ) -> None:
        self.data = data
        self.rng = rng
        self.n_neighbors = n_neighbors
        self.calibration = calibration

        # Compute nearest neighbors
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=1)
        self.knn.fit(self.data.x_train)

        # Number candidates
        self.num_cands: np.ndarray = np.count_nonzero(
            self.data.y_train, axis=1).astype(int)

    def _get_knn_y_pred(
        self, candidates: np.ndarray, is_transductive: bool,
        nn_dists: np.ndarray, nn_indices: np.ndarray,
    ) -> SplitResult:

        avg_neighbor_list: List[float] = []
        y_voting = np.zeros(
            (nn_indices.shape[0], self.data.orig_dataset.l_classes))
        for i, (nn_dist, nn_idx) in enumerate(zip(nn_dists, nn_indices)):
            # Average number of candidate labels among neighbors
            avg_neighbor_list.append(float(self.num_cands[nn_idx].mean()))

            dist_sum = nn_dist.sum()
            if dist_sum < 1e-6:
                sims = np.ones_like(nn_dist)
            else:
                sims = 1 - nn_dist / dist_sum

            for sim, idx in zip(sims, nn_idx):
                y_voting[i, :] += self.data.y_train[idx, :] * sim

        if is_transductive:
            for i in range(y_voting.shape[0]):
                y_voting[i, candidates[i] == 0] = 0

        # Scale probabilities
        prob_sum = np.sum(y_voting, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        scaled_probs = y_voting / prob_sum
        pred_list: List[int] = []
        is_sure: List[bool] = []
        guessing: List[bool] = []
        for score_row, avg_cands in zip(scaled_probs, avg_neighbor_list):
            max_val = float(np.max(score_row))
            max_idx = [
                class_lbl
                for class_lbl, score in enumerate(score_row)
                if score == max_val
            ]
            is_sure.append(max_val > self.calibration / avg_cands)
            if len(max_idx) == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            else:
                pred_list.append(int(self.rng.choice(max_idx)))
                guessing.append(True)

        # Return predictions
        return SplitResult(
            pred=np.array(pred_list),
            is_sure_pred=np.array(is_sure),
            is_guessing=np.array(guessing),
        )

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        return self._get_knn_y_pred(
            self.data.y_train, True, *self.knn.kneighbors())

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        if self.data.x_test.shape[0] == 0:
            return None

        return self._get_knn_y_pred(
            self.data.y_train, False, *self.knn.kneighbors(self.data.x_test))
