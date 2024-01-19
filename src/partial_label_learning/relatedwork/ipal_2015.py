""" Module for IPAL. """

from typing import List, Optional

import cvxpy as cp
import numpy as np
from scipy.sparse import lil_array
from sklearn.neighbors import NearestNeighbors

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class Ipal:
    """
    IPAL by Zhang and Yu,
    "Solving the Partial Label Learning Problem: An Instance-Based Approach."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        k_neighbors: int = 10,
        alpha: float = 0.95,
        max_iterations: int = 100,
    ) -> None:
        self.data = data
        self.rng = rng
        self.num_classes = self.data.orig_dataset.l_classes
        self.alpha = alpha
        self.max_iterations = max_iterations

        # Compute nearest neighbors
        num_insts = self.data.x_train.shape[0]
        self.knn = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=1)
        self.knn.fit(self.data.x_train)
        self.weight_matrix = lil_array((num_insts, num_insts), dtype=float)
        self.initial_confidence_matrix: Optional[np.ndarray] = None
        self.final_confidence_matrix: Optional[np.ndarray] = None

        # Neighborhood weight optimization problem
        num_feats = self.data.orig_dataset.m_features
        self.inst_feats = cp.Parameter(num_feats)
        self.neighbor_feats = cp.Parameter((k_neighbors, num_feats))
        self.weight_vars = cp.Variable(k_neighbors)
        constraints = [self.weight_vars >= 0]
        cost = cp.sum_squares(
            self.inst_feats - self.neighbor_feats.T @ self.weight_vars)
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def _solve_neighbor_weights_prob(
        self, inst_feats: np.ndarray, inst_neighbors: np.ndarray,
    ) -> np.ndarray:
        # Formulate optimization problem
        self.inst_feats.value = inst_feats
        self.neighbor_feats.value = np.vstack([
            self.data.x_train[j] for j in inst_neighbors
        ])
        self.prob.solve(
            solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 1})

        # Return weights
        if self.prob.status != "optimal":
            raise ValueError("Failed to find weights.")
        return self.weight_vars.value

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Compute neighbors for each instance
        nn_indices: np.ndarray = self.knn.kneighbors(return_distance=False)

        # Solve optimization problem to find weights
        for inst, inst_neighbors in enumerate(nn_indices):
            # Formulate optimization problem
            weight_vars = self._solve_neighbor_weights_prob(
                self.data.x_train[inst], inst_neighbors)

            # Store resulting weights
            for neighbor_idx, weight in zip(inst_neighbors, weight_vars):
                if float(weight) > 1e-10:
                    self.weight_matrix[neighbor_idx, inst] = float(weight)

        # Compact information and normalize
        self.weight_matrix = self.weight_matrix.tocoo()
        norm = self.weight_matrix.sum(axis=0)
        self.weight_matrix /= np.where(norm > 1e-10, norm, 1)

        # Initial labeling confidence
        num_insts = self.data.x_train.shape[0]
        initial_labeling_conf = np.zeros((num_insts, self.num_classes))
        for inst in range(num_insts):
            count_labels = np.count_nonzero(self.data.y_train[inst, :])
            initial_labeling_conf[inst, self.data.y_train[inst, :] == 1] = \
                1 / count_labels

        # Iterative propagation
        curr_labeling_conf = initial_labeling_conf.copy()
        for _ in range(self.max_iterations):
            # Propagation
            curr_labeling_conf = (
                self.alpha * self.weight_matrix.T @ curr_labeling_conf +
                (1 - self.alpha) * initial_labeling_conf
            )

            # Rescaling
            for inst in range(num_insts):
                sum_labels = np.sum(
                    curr_labeling_conf[inst, self.data.y_train[inst, :] == 1])
                curr_labeling_conf[inst, :] = np.where(
                    self.data.y_train[inst, :] == 1,
                    curr_labeling_conf[inst, :] / sum_labels,
                    0.0
                )

        # Set confidence matrices
        self.initial_confidence_matrix = initial_labeling_conf
        self.final_confidence_matrix = curr_labeling_conf

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        if self.final_confidence_matrix is None or \
                self.initial_confidence_matrix is None:
            raise ValueError("Fit must be called before predict.")

        # Compute class probability masses
        initial_class_mass: np.ndarray = np.sum(
            self.initial_confidence_matrix, axis=0)
        final_class_mass: np.ndarray = np.sum(
            self.final_confidence_matrix, axis=0)

        # Correct for imbalanced class masses
        scores = self.final_confidence_matrix.copy()
        for class_lbl in range(self.num_classes):
            if final_class_mass[class_lbl] > 1e-10:
                scores[:, class_lbl] *= initial_class_mass[class_lbl] / \
                    final_class_mass[class_lbl]

        # Scale probabilities
        prob_sum = np.sum(scores, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        scaled_probs = scores / prob_sum
        pred_list: List[int] = []
        is_sure: List[bool] = []
        guessing: List[bool] = []
        for score_row in scaled_probs:
            max_idx = np.arange(self.data.orig_dataset.l_classes)[
                score_row == np.max(score_row)]
            is_sure.append(bool(np.max(score_row) > 0.5))
            if max_idx.shape[0] == 1:
                pred_list.append(int(max_idx[0]))
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

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        if self.final_confidence_matrix is None or \
                self.initial_confidence_matrix is None:
            raise ValueError("Fit must be called before predict.")
        if self.data.x_test.shape[0] == 0:
            return None

        # Get disambiguated labels of train set
        train_res = self.get_train_pred()
        if train_res is None:
            raise ValueError("Disambiguated labels unavailable.")

        # Solve optimization problem to find weights
        nn_indices = self.knn.kneighbors(
            self.data.x_test, return_distance=False)
        scores_list: List[List[float]] = []
        for test_inst, train_inst_neighbors in enumerate(nn_indices):
            # Formulate optimization problem
            weight_vars = self._solve_neighbor_weights_prob(
                self.data.x_test[test_inst, :], train_inst_neighbors)

            # Use resulting weights
            scores_list.append([])
            for class_lbl in range(self.num_classes):
                class_vector = self.data.x_test[test_inst, :].copy()
                for train_neighbor_idx, train_neighbor_weight in zip(
                    train_inst_neighbors, weight_vars
                ):
                    if class_lbl == train_res.pred[train_neighbor_idx] and \
                            float(train_neighbor_weight) > 1e-10:
                        class_vector -= train_neighbor_weight * \
                            self.data.x_train[train_neighbor_idx]
                scores_list[-1].append(float(
                    np.dot(class_vector, class_vector)))

        # Scale probabilities
        prob = np.array(scores_list)
        prob = np.max(prob, axis=1, keepdims=True) - prob
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        scaled_probs = prob / prob_sum
        pred_list: List[int] = []
        is_sure: List[bool] = []
        guessing: List[bool] = []
        for score_row in scaled_probs:
            max_idx = np.arange(self.data.orig_dataset.l_classes)[
                score_row == np.max(score_row)]
            is_sure.append(bool(np.max(score_row) > 0.5))
            if max_idx.shape[0] == 1:
                pred_list.append(int(max_idx[0]))
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
