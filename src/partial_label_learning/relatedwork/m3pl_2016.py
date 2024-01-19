""" Module for M3PL. """

from typing import List, Optional

import cvxpy as cp
import numpy as np
from numba import njit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


@njit(cache=True, parallel=False)
def _compute_objectives(
    num_insts: int, num_classes: int, x_train: np.ndarray,
    weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    """ Calculate objectives for all instances and labels. """

    obj_i_p = np.zeros((num_insts, num_classes))
    for inst in range(num_insts):
        for class_lbl in range(num_classes):
            obj_i_p[inst, class_lbl] = (
                weights[class_lbl, :] @ x_train[inst, :]
                + biases[class_lbl]
            )
    one_minus_eta_i_p = np.zeros((num_insts, num_classes))
    for inst in range(num_insts):
        for class_lbl in range(num_classes):
            max_non_label_obj = np.max(obj_i_p[
                inst, np.arange(num_classes) != class_lbl
            ])
            one_minus_eta_i_p[inst, class_lbl] = max(
                0.0,
                1 - (obj_i_p[inst, class_lbl] - max_non_label_obj)
            )
    return one_minus_eta_i_p


class M3Pl:
    """
    M3PL by Yu and Zhang,
    "Maximum Margin Partial Label Learning."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        c_max: float = 1.0,
        reg_delta: float = 0.5,
        m_penalty: float = 1e5,
        solution_delta: float = 1e-4,
    ) -> None:
        self.data = data
        self.rng = rng
        self.c_max = c_max
        self.reg_delta = reg_delta
        self.m_penalty = m_penalty
        self.solution_delta = solution_delta

        self.num_insts = self.data.x_train.shape[0]
        self.num_features = self.data.orig_dataset.m_features
        self.num_classes = self.data.orig_dataset.l_classes

        # Model
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None

        # Class priors
        true_label_prob = 1 / np.count_nonzero(self.data.y_train, axis=1)
        num_per_class: np.ndarray = np.sum(
            self.data.y_train.T * true_label_prob,
            axis=1,
        )
        num_per_class = np.floor(num_per_class)
        num_residuals = self.num_insts - int(np.sum(num_per_class))
        if num_residuals != 0:
            add_one = np.argsort(num_per_class)[:num_residuals]
            num_per_class[add_one] += 1
        num_per_class = num_per_class.astype(int)
        if np.sum(num_per_class) != self.num_insts:
            raise ValueError("Invalid state.")

        # Ground-truth optimization problem
        self.ground_truth_scores = cp.Variable(
            (self.num_classes, self.num_insts))
        self.coefficient_matrix_param = cp.Parameter(
            (self.num_classes, self.num_insts))

        # Constraints
        constraints = [
            0 <= self.ground_truth_scores,
            self.ground_truth_scores <= 1,
            cp.sum(self.ground_truth_scores, axis=0) == 1,
            cp.sum(self.ground_truth_scores, axis=1) == num_per_class,
        ]

        # Objective
        obj = cp.Minimize(cp.sum(cp.multiply(
            self.coefficient_matrix_param, self.ground_truth_scores)))
        self.prob = cp.Problem(obj, constraints)

    def _solve_ground_truth_assignment(
        self, coefficient_matrix: np.ndarray,
    ):
        self.coefficient_matrix_param.value = coefficient_matrix.T
        self.prob.solve(
            solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 1},
            bfs=True, ignore_dpp=True,
        )
        if self.prob.status != "optimal":
            raise ValueError(f"Infeasible: {self.prob.status}")
        return np.argmax(self.ground_truth_scores.value, axis=0)

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Init regularization term
        curr_reg = 1e-5 * self.c_max

        # Init coefficient matrix
        true_label_prob = 1 / np.count_nonzero(self.data.y_train, axis=1)
        coefficient_matrix = np.zeros((self.num_insts, self.num_classes))
        for inst in range(self.num_insts):
            coefficient_matrix[inst, :] = np.where(
                self.data.y_train[inst, :] == 1,
                true_label_prob[inst],
                self.m_penalty,
            )

        # Init ground-truth label assignment
        ground_truth_label = self._solve_ground_truth_assignment(
            coefficient_matrix)

        # Gradually increase regularization
        while curr_reg < self.c_max:
            curr_reg = min(self.c_max, (1 + self.reg_delta) * curr_reg)
            prev_obj_val = np.inf
            curr_obj_val = np.inf

            # Alternating optimization
            while (
                np.isinf(prev_obj_val) or
                prev_obj_val - curr_obj_val >= self.solution_delta
            ):
                # Update objective
                prev_obj_val = curr_obj_val

                # Fit model according to current ground-truth labels
                op3_solver = OneVsRestClassifier(LinearSVC(
                    loss="squared_hinge", dual="auto",
                    C=curr_reg, fit_intercept=True,
                ))
                op3_solver.fit(self.data.x_train, ground_truth_label)
                weights = np.vstack([
                    op3_solver.estimators_[
                        np.where(op3_solver.classes_ == class_lbl)[0][0]
                    ].coef_[0]
                    if class_lbl in op3_solver.classes_
                    else np.zeros(self.num_features)
                    for class_lbl in range(self.num_classes)
                ])
                biases = np.array([
                    op3_solver.estimators_[
                        np.where(op3_solver.classes_ == class_lbl)[0][0]
                    ].intercept_[0]
                    if class_lbl in op3_solver.classes_
                    else 0.0
                    for class_lbl in range(self.num_classes)
                ])

                # Calculate objectives for all instances and labels.
                one_minus_eta_i_p = _compute_objectives(
                    self.num_insts, self.num_classes,
                    self.data.x_train, weights, biases,
                )

                # Update coefficient matrix
                coefficient_matrix = np.where(
                    self.data.y_train == 1,
                    one_minus_eta_i_p,
                    self.m_penalty,
                )

                # Update ground-truth label assignment
                ground_truth_label = self._solve_ground_truth_assignment(
                    coefficient_matrix)

                # Update global objective value
                curr_obj_val = 0.0
                for class_lbl in range(self.num_classes):
                    curr_obj_val += 0.5 * np.dot(
                        weights[class_lbl, :], weights[class_lbl, :])
                for inst in range(self.num_insts):
                    curr_obj_val += curr_reg * one_minus_eta_i_p[
                        inst, ground_truth_label[inst]]

        # Set model
        self.weights = weights
        self.biases = biases

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if self.weights is None or self.biases is None:
            return None
        if data.shape[0] == 0:
            return None

        scores = np.zeros((data.shape[0], self.num_classes))
        for class_lbl in range(self.num_classes):
            scores[:, class_lbl] = (
                data @ self.weights[class_lbl, :]
                + self.biases[class_lbl]
            )

        # If in transductive setting, restrict selection to candidate labels
        if is_transductive:
            for i in range(data.shape[0]):
                for j in range(self.num_classes):
                    if candidates[i, j] == 0:
                        scores[i, j] = -np.inf

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
