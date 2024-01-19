""" Module for CLPL. """

from typing import List, Optional

import numpy as np
from sklearn.svm import LinearSVC

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class Clpl:
    """
    CLPL by Cour, Sapp, and Taskar,
    "Learning from Partial Labels."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
    ) -> None:
        self.data = data
        self.rng = rng

        self.num_features = self.data.orig_dataset.m_features
        self.num_classes = self.data.orig_dataset.l_classes

        # Model
        self.model: Optional[LinearSVC] = None

    def _single_feature_stack(self, x_i: np.ndarray, y_i: int) -> np.ndarray:
        res = np.zeros(self.num_features * self.num_classes)
        res[y_i * self.num_features:(y_i + 1) * self.num_features] = x_i
        return res

    def _feature_stack(self, x_i: np.ndarray, ys_i: np.ndarray) -> np.ndarray:
        res = np.zeros(self.num_features * self.num_classes)
        for i, y_cand in enumerate(ys_i):
            if y_cand == 1:
                res[i * self.num_features:(i + 1) * self.num_features] = x_i
        res = res / np.count_nonzero(ys_i)
        return res

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Construct positively labeled examples
        feats_list = []
        targets_list = []
        for x_i, y_i in zip(self.data.x_train, self.data.y_train):
            feats_list.append(self._feature_stack(x_i, y_i))
            targets_list.append(1)

        # Construct negatively labeled examples
        for x_i, y_i in zip(self.data.x_train, self.data.y_train):
            for compl_class_lbl in range(self.num_classes):
                if y_i[compl_class_lbl] == 0:
                    feats_list.append(
                        self._single_feature_stack(x_i, compl_class_lbl))
                    targets_list.append(0)

        # Shuffle
        perm = self.rng.permutation(len(feats_list))
        feats = np.vstack(feats_list)[perm].copy()
        targets = np.array(targets_list)[perm].copy()

        # Fit support-vector machine
        self.model = LinearSVC(
            random_state=self.rng.integers(int(1e6)),
            loss="squared_hinge", dual="auto",
        )
        self.model.fit(feats, targets)

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if self.model is None:
            return None
        if data.shape[0] == 0:
            return None

        feats_list = []
        for x_i in data:
            for j in range(self.num_classes):
                feats_list.append(self._single_feature_stack(x_i, j))
        feats = np.vstack(feats_list)
        scores = self.model.decision_function(feats).reshape(
            (data.shape[0], self.num_classes))

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
