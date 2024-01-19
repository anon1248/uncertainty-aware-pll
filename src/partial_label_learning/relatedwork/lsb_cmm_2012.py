"""
Module for LSB-CMM.

Translation from the Matlab code of
https://github.com/xryanyjy/LSB_CMM_Matlab.
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, gammaln

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


@njit(cache=True, parallel=False)
def _decompose_expectation(
    log_phi: np.ndarray, e_log: np.ndarray, y_train: np.ndarray,
    k_components: int, num_insts: int, num_classes: int,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """ Decomposes the expectation term.

    Args:
        log_phi (np.ndarray): A (num_insts, k_components) array.
        e_log (np.ndarray): A (num_classes, k_components) array.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]:
        The decomposed results.
    """

    res_logpy = 0.0
    res_entropy = 0.0
    res_e_phi = np.zeros((k_components, num_insts))
    res_count = np.zeros((k_components, num_classes))

    for inst in range(num_insts):
        cands = np.where(y_train[inst, :] == 1)[0]
        num_cands = cands.shape[0]
        ins_e_phi = np.zeros((num_cands, k_components))

        for cand_idx in range(num_cands):
            lab = cands[cand_idx]
            for k in range(k_components):
                ins_e_phi[cand_idx, k] = log_phi[inst, k] + e_log[lab, k]

        # Log-normalize
        ins_e_phi = ins_e_phi - np.max(ins_e_phi)
        ins_e_phi = np.exp(ins_e_phi)
        ins_e_phi = ins_e_phi / np.sum(ins_e_phi)

        for cand_idx in range(num_cands):
            lab = cands[cand_idx]
            for k in range(k_components):
                res_e_phi[k, inst] += ins_e_phi[cand_idx, k]
                res_count[k, lab] += ins_e_phi[cand_idx, k]

                if ins_e_phi[cand_idx, k] > 1e-20:
                    res_entropy -= (
                        ins_e_phi[cand_idx, k] *
                        np.log(ins_e_phi[cand_idx, k])
                    )
                    res_logpy += ins_e_phi[cand_idx, k] * e_log[lab, k]

    return res_logpy, res_entropy, res_e_phi, res_count


@njit(cache=True, parallel=False)
def _objective_value(
    weights: np.ndarray, data: np.ndarray,
    phik: np.ndarray, phis: np.ndarray,
    sigma_squ: float,
) -> np.ndarray:
    obj_val = (weights[1:] @ weights[1:]) / (2 * sigma_squ)
    value = data @ weights
    obj_val = (
        obj_val +
        (phik + phis) @ np.log(1 + np.exp(value)) -
        (phik @ value)
    )
    return obj_val


@njit(cache=True, parallel=False)
def _objective_grad(
    weights: np.ndarray, data: np.ndarray,
    phik: np.ndarray, phis: np.ndarray,
    sigma_squ: float,
) -> np.ndarray:
    obj_grad = weights / sigma_squ
    value = data @ weights
    obj_grad[0] = 0
    obj_grad = obj_grad + (
        data.T @ (np.exp(value) / (1 + np.exp(value)) * (phik + phis) - phik)
    )
    return obj_grad


class LsbCmm:
    """
    LSB-CMM by Liu and Dietterich,
    "A Conditional Multinomial Mixture Model for Superset Label Learning."

    Translation from the Matlab code of
    https://github.com/xryanyjy/LSB_CMM_Matlab.
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        k_comp_factor: int = 5,
        alpha: float = 0.05,
        sigma_squ: float = 1.0,
        max_iterations: int = 100,
    ) -> None:
        self.data = data
        self.rng = rng
        self.num_classes = self.data.orig_dataset.l_classes
        self.num_insts = self.data.x_train.shape[0]

        # Parameters
        self.k_components = k_comp_factor * self.num_classes
        self.alpha = alpha
        self.sigma_squ = sigma_squ
        self.max_iterations = max_iterations

        # Model
        self.weights: Optional[np.ndarray] = None
        self.theta: Optional[np.ndarray] = None
        self.likelihood: Optional[float] = None

    def _ovr_code(self) -> np.ndarray:
        """ Generates the one-vs-rest mapping of the individual components.

        Returns:
            np.ndarray: The mapping of classes to components.
        """

        code = np.zeros((self.num_classes, self.k_components))
        cls_of_comp = np.arange(self.k_components) % self.num_classes

        for component in range(self.k_components):
            code[cls_of_comp[component], component] = 1

        code = code * 10 + 1
        return code

    def _expectation_log_theta(self, ealpha: np.ndarray) -> np.ndarray:
        """
        Returns the expected value of log theta
        with theta being a Dirichlet distribution.

        Args:
            ealpha (np.ndarray): The parameter of the Dirichlet distribution.

        Raises:
            ValueError: If an invalid value is encountered.

        Returns:
            np.ndarray: The calculated expectation.
        """

        mean_log = ealpha.copy()
        for comp in range(ealpha.shape[1]):
            dirich_sample = self.rng.dirichlet(ealpha[:, comp], 1000)
            dirich_sample = np.clip(dirich_sample, 1e-100, 1)
            mean_log[:, comp] = np.mean(np.log(dirich_sample), axis=0)
        return mean_log

    def _ldb_cmm_log_norm(self, log_prob: np.ndarray) -> np.ndarray:
        """ Normalizes each row to be the logarithm of a multinomial probability.

        Args:
            log_prob (np.ndarray): The log probability.

        Returns:
            np.ndarray: The normalized data.
        """

        max_val = np.max(log_prob, axis=1, keepdims=True)
        new_log_prob = log_prob.copy() - max_val
        prob = np.exp(new_log_prob)
        norm = np.sum(prob, axis=1, keepdims=True)
        return new_log_prob - np.log(norm)

    def _lsb_cmm_log_stick_break(
        self, data: np.ndarray, weights: np.ndarray,
    ) -> np.ndarray:
        """ Stick breaking process with with logrithm values.

        Args:
            data (np.ndarray): Covariates of the data.
            weights (np.ndarray): Weights for logistic function.

        Returns:
            np.ndarray: Multinomial probabilities of choosing each component.
        """

        link: np.ndarray = data @ weights
        log_v1 = -np.log(1 + np.exp(-link))

        for i in range(log_v1.shape[0]):
            for j in range(log_v1.shape[1]):
                if -link[i, j] > 30:
                    log_v1[i, j] = link[i, j]

        log_v0 = -np.log(1 + np.exp(link))
        for i in range(log_v0.shape[0]):
            for j in range(log_v0.shape[1]):
                if link[i, j] > 30:
                    log_v0[i, j] = -link[i, j]

        accum = log_v0.copy()
        accum[:, 0] = 0

        for k in range(1, log_v1.shape[1]):
            accum[:, k] = accum[:, k - 1] + log_v0[:, k - 1]

        log_phi = log_v1 + accum
        log_phi = self._ldb_cmm_log_norm(log_phi)

        return log_phi

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Init model
        x_data = np.hstack((
            np.ones((self.num_insts, 1)),
            self.data.x_train.copy(),
        ))
        num_dims = x_data.shape[1]
        weights = np.zeros((num_dims, self.k_components))
        weights[0, -1] = np.inf
        ealpha = self._ovr_code()

        log_phi = np.zeros((self.num_insts, self.k_components))
        likelihood = np.zeros(self.max_iterations)

        for epoch in range(self.max_iterations):
            # E-step
            elog = self._expectation_log_theta(ealpha)
            logpy, entropy, ephi, count = _decompose_expectation(
                log_phi, elog, self.data.y_train, self.k_components,
                self.num_insts, self.num_classes,
            )
            count = count.T
            ealpha = count + self.alpha
            theta = ealpha / np.sum(ealpha, axis=0)

            # M step
            update_completed = False
            for k in range(self.k_components - 1):
                phik = ephi[k, :]
                phis = np.sum(ephi[(k + 1):, :], axis=0)
                result: OptimizeResult = minimize(
                    _objective_value, weights[:, k],
                    args=(x_data, phis, phik, self.sigma_squ),
                    method="L-BFGS-B",
                    jac=_objective_grad,
                )
                if not result.success:
                    break
                weights[:, k] = -result.x
            else:
                update_completed = True
            if not update_completed:
                break

            # Calculate the lower bound to decide when to stop
            log_phi = self._lsb_cmm_log_stick_break(x_data, weights)
            elog = self._expectation_log_theta(ealpha)

            sum_matrix1 = np.sum(ephi.T * log_phi)
            sum_matrix2 = gammaln(np.sum(ealpha))
            sum_matrix3 = np.sum(gammaln(ealpha))
            sum_matrix4 = np.sum((ealpha - self.alpha) * elog)
            sum_matrix5 = np.sum(weights[1:, :-1] * weights[1:, :-1])

            term1 = sum_matrix1
            term2 = logpy
            term3 = entropy
            term4 = sum_matrix2 - sum_matrix3 - self.k_components * \
                gammaln(self.alpha * self.num_classes) + self.k_components * \
                (gammaln(self.alpha) * self.num_classes) + sum_matrix4
            term5 = -0.5 * sum_matrix5 / self.sigma_squ

            likelihood[epoch] = term1 + term2 + term3 - term4 + term5

            if epoch > 10:
                matrix9 = likelihood[(epoch - 2):(epoch + 1)]
                curr_likeli = np.mean(matrix9)
                matrix10 = likelihood[(epoch - 5):(epoch - 2)]
                prev_likeli = np.mean(matrix10)
                if (curr_likeli - prev_likeli) < 1e-4 * np.abs(curr_likeli):
                    break

        self.weights = weights
        self.theta = theta
        self.likelihood = likelihood[-1]

    def _lsb_cmm_stick_break(self, probs: np.ndarray) -> np.ndarray:
        phi = probs.copy()
        temp = (1 - probs).copy()
        for k in range(1, probs.shape[1]):
            for i in range(probs.shape[0]):
                phi[i, k] = temp[i, k-1] * probs[i, k]
                temp[i, k] = temp[i, k-1] * (1 - probs[i, k])
        return phi

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if self.weights is None or self.theta is None:
            return None
        if data.shape[0] == 0:
            return None

        temp_col = np.ones((data.shape[0], 1))
        data_with_temp = np.hstack((temp_col, data.copy()))
        probs = expit(data_with_temp @ self.weights)
        phi = self._lsb_cmm_stick_break(probs)
        prob = phi @ self.theta.T

        # If in transductive setting, restrict selection to candidate labels
        if is_transductive:
            for i in range(data.shape[0]):
                for j in range(self.num_classes):
                    if candidates[i, j] == 0:
                        prob[i, j] = 0

        # Scale probabilities
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

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        res = self._predict(self.data.x_train, self.data.y_train, True)
        if res is None:
            raise ValueError("Result cannot be None.")
        return res

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        return self._predict(self.data.x_test, self.data.y_test, False)
