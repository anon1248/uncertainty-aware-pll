""" Module for one-vs-rest partial label learning. """

from typing import List, Optional

import numpy as np
from sklearn.base import clone

from partial_label_learning.baselines.constant_clf import ConstantClassifier
from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class OvrPll:
    """ Simple one-vs-rest partial label learning baseline. """

    def __init__(
        self, data: Datasplit, rng: np.random.Generator, binary_clf,
    ) -> None:
        self.data = data
        self.rng = rng
        self.binary_clf = binary_clf
        self.num_classes = self.data.orig_dataset.l_classes
        self.binary_clfs = []

        # Fit binary classifiers
        for class_lbl in range(self.num_classes):
            if np.unique(self.data.y_train[:, class_lbl]).shape[0] == 2:
                class_clf = clone(binary_clf)
                class_clf.fit(
                    self.data.x_train, self.data.y_train[:, class_lbl])
                self.binary_clfs.append(class_clf)
            else:
                self.binary_clfs.append(ConstantClassifier(-np.inf))

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if data.shape[0] == 0:
            return None

        # Compute probs per class
        class_probs = np.hstack([
            clf.decision_function(data).reshape(-1, 1)
            for clf in self.binary_clfs
        ])

        # Eliminate non-candidates if in transductive setting
        if is_transductive:
            for i in range(data.shape[0]):
                for j in range(self.num_classes):
                    if candidates[i, j] == 0:
                        class_probs[i, j] = -np.inf

        # Extract predictions
        pred_list: List[int] = []
        is_sure: List[bool] = []
        guessing: List[bool] = []
        for score_row in class_probs:
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
