""" Module for PL-SVM. """

import math
from typing import List, Optional

import numpy as np
from numba import njit

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class WeightVector:
    """ Wraps a weight vector. """

    def __init__(self, m_features: int, l_classes: int) -> None:
        self.m_features = m_features
        self.l_classes = l_classes
        self.n_dims = m_features * l_classes
        self.weights = np.zeros(self.n_dims, dtype=float)

    def norm(self) -> float:
        """ Returns the norm of the weight vector.

        Returns:
            float: The norm.
        """

        return float(np.linalg.norm(self.weights, 2))

    def scale(self, scale: float) -> None:
        """ Scales the vector.

        Args:
            scale (float): Scales the vector.
        """

        self.weights *= scale

    def add_phi_xy(self, scale: float, x_i: np.ndarray, y_i: int) -> None:
        """ Add a multiple of Phi(x, y) to the weight vector.

        Args:
            scale (float): The scale.
            x_i (np.ndarray): The features.
            y_i (int): The candidate label.
        """

        self.weights[
            self.m_features * y_i:self.m_features * (y_i + 1)
        ] += scale * x_i


@njit(cache=True, parallel=False)
def _wt_phi_xy(
    weights: np.ndarray, x_i: np.ndarray, y_i: int, m_features: int,
) -> float:
    """ Computes w^T * Phi(x, y).

    Args:
        x_i (np.ndarray): The features.
        y_i (int): The candidate label.

    Returns:
        float: The result.
    """

    return np.sum(
        weights[m_features * y_i:m_features * (y_i + 1)]
        * x_i
    )


class PlSvm:
    """
    PL-SVM by Nguyen and Caruana,
    "Classification with Partial Labels."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        max_iterations: int = 100000,
        lambda_reg: float = 1.0,
    ) -> None:
        self.data = data
        self.rng = rng
        self.max_iterations = max_iterations
        self.lambda_reg = lambda_reg
        self.num_classes = self.data.orig_dataset.l_classes

        # Model
        self.weight_vector: Optional[WeightVector] = None

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Init weight vector
        self.weight_vector = WeightVector(
            self.data.orig_dataset.m_features, self.num_classes)
        num_insts = self.data.x_train.shape[0]

        # Stochastic training loop
        for epoch in range(self.max_iterations):
            # Pick random element
            inst = self.rng.choice(num_insts)
            x_i = self.data.x_train[inst]
            ys_i = self.data.y_train[inst]

            # Compute max margin
            pos_scores = [
                _wt_phi_xy(
                    self.weight_vector.weights, x_i,
                    class_lbl, self.weight_vector.m_features,
                )
                if ys_i[class_lbl] == 1 else -np.inf
                for class_lbl in range(self.num_classes)
            ]
            max_pos_class = int(np.argmax(pos_scores))
            neg_scores = [
                _wt_phi_xy(
                    self.weight_vector.weights, x_i,
                    class_lbl, self.weight_vector.m_features,
                )
                if ys_i[class_lbl] == 0 else -np.inf
                for class_lbl in range(self.num_classes)
            ]
            max_neg_class = int(np.argmax(neg_scores))

            # Compute eta
            eta = 1 / (self.lambda_reg * (epoch + 1))
            weight_scaling = max(1e-9, 1 - eta * self.lambda_reg)

            # Regularize weight
            self.weight_vector.scale(weight_scaling)

            # Add feedback from violations
            if pos_scores[max_pos_class] - neg_scores[max_neg_class] < 1:
                self.weight_vector.add_phi_xy(eta, x_i, max_pos_class)
                self.weight_vector.add_phi_xy(-eta, x_i, max_neg_class)

            # Normalize vector
            w_norm = self.weight_vector.norm()
            if w_norm > 1e-10:
                projection = 1 / (math.sqrt(self.lambda_reg) * w_norm)
                if projection < 1:
                    self.weight_vector.scale(projection)

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if not self.weight_vector:
            return None
        if data.shape[0] == 0:
            return None

        scores = -np.inf * np.ones((data.shape[0], self.num_classes))
        for i, x_i in enumerate(data):
            for j in range(self.num_classes):
                # If in transductive setting, only assign non-zero score for candidate labels
                if not is_transductive or candidates[i, j] == 1:
                    scores[i, j] = _wt_phi_xy(
                        self.weight_vector.weights, x_i,
                        j, self.weight_vector.m_features,
                    )

        # Extract predictions
        pred_list: List[int] = []
        is_sure: List[bool] = []
        guessing: List[bool] = []
        for score_row in scores:
            is_sure.append(bool(np.count_nonzero(score_row > 0) == 1))
            max_idx = np.arange(self.num_classes)[
                score_row == np.max(score_row)]
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

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        res = self._predict(self.data.x_train, self.data.y_train, True)
        if res is None:
            raise ValueError("Result must exist.")
        return res

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        return self._predict(self.data.x_test, self.data.y_test, False)
