""" Module for PL-ECOC. """

import math
from typing import List, Optional, Set

import numpy as np
from sklearn.svm import SVC

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class PlEcoc:
    """
    PL-ECOC by Zhang, Yu, and Tang,
    "Disambiguation-Free Partial Label Learning."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
    ) -> None:
        self.data = data
        self.rng = rng
        self.num_insts = self.data.x_train.shape[0]
        self.num_classes = self.data.orig_dataset.l_classes
        self.codeword_length = int(math.ceil(10 * np.log2(self.num_classes)))

        # Encoded candidates
        self.enc_multiplier: List[int] = [
            2 ** code_pos for code_pos in range(self.num_classes)
        ]
        self.y_train_enc = [
            sum(
                self.enc_multiplier[code_pos] * int(y_row[code_pos])
                for code_pos in range(self.num_classes)
            ) for y_row in self.data.y_train
        ]
        self.num_classes_mask = sum(self.enc_multiplier)
        self.min_binary_train_size = max(1, self.num_insts // 10)

        # Model
        self.model_is_fit = False
        self.coding_matrix: np.ndarray = np.array([])
        self.perf_matrix: np.ndarray = np.array([])
        self.binary_clfs: List[SVC] = []

    def _compute_coding_column(self) -> bool:
        # If less then 10 classes, exhaustive search
        possible_codes = []
        if self.num_classes <= 10:
            for pos_set in range(1, (self.num_classes_mask >> 1) + 1):
                # Define positive and negative class labels
                neg_set = self.num_classes_mask - pos_set

                # Count training instances
                num_pos = 0
                num_neg = 0
                for cand in self.y_train_enc:
                    if (cand & pos_set) == cand:
                        num_pos += 1
                    elif (cand & neg_set) == cand:
                        num_neg += 1

                # Check if code provides valid dichotomy
                if (
                    num_pos and num_neg and
                    num_pos + num_neg >= self.min_binary_train_size
                ):
                    possible_codes.append(pos_set)

        # Else, random choice
        else:
            max_retries = 10000
            retries = 0
            already_used: Set[int] = set()

            while (
                retries < max_retries and
                len(possible_codes) < self.codeword_length
            ):
                # Define positive and negative class labels
                pos_set_list = list(map(int, list(self.rng.choice(
                    2, size=self.num_classes-1, replace=True))))
                pos_set_list.append(0)
                if all(item == 0 for item in pos_set_list):
                    retries += 1
                    continue
                pos_set = sum(
                    self.enc_multiplier[i] * pos_set_list[i]
                    for i in range(self.num_classes)
                )

                if pos_set in already_used:
                    retries += 1
                    continue
                neg_set = self.num_classes_mask - pos_set

                # Count training instances
                num_pos = 0
                num_neg = 0
                for cand in self.y_train_enc:
                    if (cand & pos_set) == cand:
                        num_pos += 1
                    elif (cand & neg_set) == cand:
                        num_neg += 1

                # Check if code provides valid dichotomy
                if (
                    num_pos and num_neg and
                    num_pos + num_neg >= self.min_binary_train_size
                ):
                    retries = 0
                    possible_codes.append(pos_set)
                    already_used.add(pos_set)
                else:
                    retries += 1

        # Sample from possible codes
        final_codeword_length = min(self.codeword_length, len(possible_codes))
        if not final_codeword_length:
            return False
        codes_picked: List[int] = sorted(map(int, list(self.rng.choice(
            possible_codes, size=final_codeword_length,
            replace=False, shuffle=False,
        ))))

        # Extract coding matrix
        self.coding_matrix = self.coding_matrix[
            :self.num_classes, :len(codes_picked)
        ]
        self.codeword_length = len(codes_picked)
        for i, code in enumerate(codes_picked):
            for j, bit in enumerate(list(reversed(f"{code:b}"))):
                self.coding_matrix[j, i] = 1 if bit == "1" else -1
        return True

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Compute coding matrix
        self.coding_matrix = -np.ones(
            (self.num_classes, self.codeword_length), dtype=int)
        self.binary_clfs = []
        if not self._compute_coding_column():
            return

        for codeword_idx in range(self.codeword_length):
            # q-bits column coding
            column_coding = self.coding_matrix[:, codeword_idx]

            # Derive training sets
            pos_set = int(np.sum(
                np.where(column_coding == 1, 1, 0) * self.enc_multiplier))
            neg_set = self.num_classes_mask - pos_set
            data_mask = np.zeros(self.num_insts, dtype=bool)
            contains_positive = False
            contains_negative = False
            targets = []
            for i, cand in enumerate(self.y_train_enc):
                if (cand & pos_set) == cand:
                    data_mask[i] = True
                    targets.append(True)
                    contains_positive = True
                elif (cand & neg_set) == cand:
                    data_mask[i] = True
                    targets.append(False)
                    contains_negative = True

            # Found dichotomy with enough training instances
            if contains_positive and contains_negative:
                # Train binary classifier on dichotomy
                clf = SVC(random_state=self.rng.integers(int(1e6)))
                clf.fit(self.data.x_train[data_mask, :], targets)
                self.binary_clfs.append(clf)
            else:
                # Invalid state
                raise RuntimeError("Invalid state.")

        # Pre-compute binary classifier predictions
        pred_results = [
            self.binary_clfs[codeword_idx].predict(self.data.x_train)
            for codeword_idx in range(self.codeword_length)
        ]

        # Compute performance matrix
        self.perf_matrix = np.zeros(
            (self.num_classes, self.codeword_length), dtype=float)
        for class_idx in range(self.num_classes):
            for codeword_idx in range(self.codeword_length):
                mask = self.data.y_train[:, class_idx] == 1
                mask_norm = np.count_nonzero(mask)
                self.perf_matrix[class_idx, codeword_idx] = (
                    np.count_nonzero(
                        pred_results[codeword_idx][mask]
                        == (self.coding_matrix[class_idx, codeword_idx] == 1)
                    )
                    / (mask_norm if mask_norm != 0 else 1)
                )

        # Normalize performance matrix
        norm = self.perf_matrix.sum(axis=1)
        self.perf_matrix = (
            self.perf_matrix.transpose() / np.where(norm < 1e-6, 1, norm)
        ).transpose()
        self.model_is_fit = True

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if not self.model_is_fit:
            return None
        if data.shape[0] == 0:
            return None

        # Precompute all decision function outputs
        decision_func_outputs = np.vstack([
            self.binary_clfs[codeword_idx].decision_function(data)
            for codeword_idx in range(self.codeword_length)
        ])

        scores_list: List[np.ndarray] = []
        for inst, y_row in enumerate(candidates):
            class_scores = np.zeros(self.num_classes)
            for class_idx in range(self.num_classes):
                if is_transductive and y_row[class_idx] != 1:
                    continue
                for codeword_idx in range(self.codeword_length):
                    perf_entry = self.perf_matrix[class_idx, codeword_idx]
                    if perf_entry == 0.0:
                        continue
                    pred = float(decision_func_outputs[codeword_idx, inst])
                    coding_entry = self.coding_matrix[class_idx, codeword_idx]
                    class_scores[class_idx] += (
                        perf_entry * np.exp(pred * coding_entry)
                    )
            scores_list.append(class_scores)

        # Extract predictions
        prob = np.array(scores_list)
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
            raise ValueError("Result must exist.")
        return res

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        return self._predict(self.data.x_test, self.data.y_test, False)
