""" Module for SURE. """

from typing import List, Optional

import cvxpy as cp
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class Sure:
    """
    SURE by Feng and An,
    "Partial Label Learning with Self-Guided Retraining."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        lambda_reg: float = 0.3,
        beta_reg: float = 0.05,
        max_iterations: int = 100,
        convergence_delta: float = 0.1,
    ) -> None:
        self.data = data
        self.rng = rng
        self.lambda_reg = lambda_reg
        self.beta_reg = beta_reg
        self.max_iterations = max_iterations
        self.convergence_delta = convergence_delta

        self.num_insts = self.data.x_train.shape[0]
        self.num_feats = self.data.orig_dataset.m_features
        self.num_classes = self.data.orig_dataset.l_classes

        # Compute parameter of kernel: Avg. distance
        dists = 0.0
        if self.num_insts * self.num_insts / 2 > 10000:
            count = 10000
            for inst1, inst2 in self.rng.choice(
                self.num_insts, size=(10000, 2), replace=True
            ):
                dists += float(np.linalg.norm(
                    self.data.x_train[inst1, :] -
                    self.data.x_train[inst2, :]
                ))
        else:
            count = 0
            for inst1 in range(self.num_insts - 1):
                for inst2 in range(inst1 + 1, self.num_insts):
                    count += 1
                    dists += float(np.linalg.norm(
                        self.data.x_train[inst1, :] -
                        self.data.x_train[inst2, :]
                    ))
        self.rbf_gamma = 1 / (2 * ((dists / count) ** 2))

        # Model
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None

        # QP base problem
        self.labeling_vars = cp.Variable(self.num_classes)
        self.max_labeling_var_ind_param = cp.Parameter(self.num_classes)
        self.cand_mask_param = cp.Parameter(self.num_classes)
        self.scores_param = cp.Parameter(self.num_classes)
        constraints = [
            0 <= self.labeling_vars,
            self.labeling_vars <= 1,
            cp.sum(self.labeling_vars) == 1,
            self.labeling_vars <= self.cand_mask_param,
            self.labeling_vars <= (
                self.max_labeling_var_ind_param @ self.labeling_vars
            ),
        ]
        obj_val = (
            cp.sum_squares(self.labeling_vars - self.scores_param) -
            self.lambda_reg * (
                self.max_labeling_var_ind_param @ self.labeling_vars
            )
        )
        self.qp_prob = cp.Problem(cp.Minimize(obj_val), constraints)

    def _solve_qp_problem(self, inst: int, scores_row: np.ndarray) -> np.ndarray:
        # If only one candidate, return
        if np.count_nonzero(self.data.y_train[inst, :]) == 1:
            return self.data.y_train[inst, :].copy()

        # Set candidate mask and scores
        self.cand_mask_param.value = self.data.y_train[inst, :]
        self.scores_param.value = np.where(
            self.data.y_train[inst, :] == 1,
            scores_row, 0,
        )

        # Identify maximum score variable
        max_q_label = np.argmax(np.where(
            self.data.y_train[inst, :] == 1,
            scores_row, -np.inf,
        ))
        max_q_ind = np.zeros(self.num_classes, dtype=int)
        max_q_ind[max_q_label] = 1
        self.max_labeling_var_ind_param.value = max_q_ind

        # Solve problem
        self.qp_prob.solve(
            solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 1})
        if self.qp_prob.status != "optimal":
            raise ValueError(f"Infeasible: {self.qp_prob.status}")
        return np.where(
            self.data.y_train[inst, :] == 1,
            self.labeling_vars.value, 0.0,
        )

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Compute kernel
        kernel = rbf_kernel(self.data.x_train, gamma=self.rbf_gamma)
        one_vector = np.ones(self.num_insts)
        one_matrix = np.ones((self.num_insts, self.num_insts))
        beta_id = self.beta_reg * np.identity(self.num_insts)

        # Init labeling matrix
        prev_labeling_matrix: np.ndarray = (
            self.data.y_train.astype(float) /
            np.count_nonzero(self.data.y_train, axis=1, keepdims=True)
        )
        curr_labeling_matrix = prev_labeling_matrix.copy()

        # Main training loop
        for _ in range(self.max_iterations):
            # Update weights and biases
            weights = np.linalg.inv(
                kernel + beta_id -
                (1 / self.num_insts) *
                (one_matrix @ kernel)
            ) @ (
                curr_labeling_matrix -
                (1 / self.num_insts) *
                (one_matrix @ curr_labeling_matrix)
            )
            biases = (1 / self.num_insts) * (
                curr_labeling_matrix.T @ one_vector -
                weights.T @ kernel.T @ one_vector
            )
            scores = kernel @ weights + \
                one_vector.reshape(-1, 1) @ biases.reshape(1, -1)

            # Compute new pseudo-labeling matrix through optimization
            for inst in range(self.num_insts):
                curr_labeling_matrix[inst, :] = self._solve_qp_problem(
                    inst, scores[inst, :])

            # Normalize
            curr_labeling_matrix = curr_labeling_matrix / np.sum(
                curr_labeling_matrix, axis=1, keepdims=True)

            # Check convergence
            if np.linalg.norm(
                curr_labeling_matrix - prev_labeling_matrix, "fro"
            ) < self.convergence_delta:
                break

            # Make current matrix the previous one
            prev_labeling_matrix = curr_labeling_matrix.copy()

        # Set weights and biases
        self.weights = weights
        self.biases = biases

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if self.weights is None or self.biases is None:
            return None
        if data.shape[0] == 0:
            return None

        # Compute kernel between train instances and data
        kernel_with_train = rbf_kernel(
            data, self.data.x_train, gamma=self.rbf_gamma)

        # Compute scores
        scores = kernel_with_train @ self.weights + self.biases

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
