""" Module for classifier trained with full ground-truth. """

from typing import List, Optional

import numpy as np
from sklearn.multiclass import OneVsRestClassifier

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class FullySupervisedClf:
    """ Classifier trained with full ground-truth. """

    def __init__(
        self, data: Datasplit, rng: np.random.Generator, clf,
    ) -> None:
        self.data = data
        self.rng = rng
        self.clf = OneVsRestClassifier(clf, n_jobs=1)
        self.num_classes = self.data.orig_dataset.l_classes
        self.clf.fit(self.data.x_train, self.data.y_true_train)

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if data.shape[0] == 0:
            return None

        # Compute probs per class
        class_probs = -np.inf * np.ones((data.shape[0], self.num_classes))
        class_probs[:, self.clf.classes_] = self.clf.decision_function(data)

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
