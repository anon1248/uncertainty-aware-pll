""" Module for PALOC. """

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.svm import SVC

from partial_label_learning.baselines.constant_clf import ConstantClassifier
from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class Paloc:
    """
    PALOC by Wu and Zhang,
    "Towards Enabling Binary Decomposition for Partial Label Learning."
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        mu_param: float = 10.0,
    ) -> None:
        self.data = data
        self.rng = rng
        self.mu_param = mu_param
        self.num_insts = self.data.x_train.shape[0]
        self.num_classes = self.data.orig_dataset.l_classes

        # Model
        self.ovo_clfs: Optional[Dict[
            Tuple[int, int],
            Union[SVC, ConstantClassifier]
        ]] = None
        self.ovr_clfs: Optional[List[Union[SVC, ConstantClassifier]]] = None

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Compute one-vs-one classifiers
        self.ovo_clfs = {}
        clf_preds = {}
        clf_preds_ordered = [self.data.x_train.copy()]
        targets = np.zeros(self.num_insts, dtype=int)
        y_masks = self.data.y_train.astype(bool)
        for class1 in range(self.num_classes - 1):
            for class2 in range(class1 + 1, self.num_classes):
                # Find examples
                is_class1_cand = y_masks[:, class1]
                is_class2_cand = y_masks[:, class2]
                pos_set = is_class1_cand & ~is_class2_cand
                targets[pos_set] = 1
                neg_set = ~is_class1_cand & is_class2_cand
                targets[neg_set] = -1
                in_train_set = pos_set | neg_set

                if np.unique(targets[in_train_set]).shape[0] == 2:
                    # Train SVC
                    clf = SVC(random_state=self.rng.integers(int(1e6)))
                    clf.fit(
                        self.data.x_train[in_train_set],
                        targets[in_train_set],
                    )
                    self.ovo_clfs[class1, class2] = clf
                    clf_preds[class1, class2] = clf.decision_function(
                        self.data.x_train)
                else:
                    # Not able to distinguish classes
                    self.ovo_clfs[class1, class2] = ConstantClassifier()
                    clf_preds[class1, class2] = np.zeros(self.num_insts)
                clf_preds_ordered.append(
                    clf_preds[class1, class2].reshape(-1, 1))

        # Disambiguate partial labels
        ovo_counts = np.zeros((self.num_insts, self.num_classes), dtype=int)
        for class_lbl in range(self.num_classes):
            for class1 in range(class_lbl):
                ovo_counts[:, class_lbl] += np.where(
                    clf_preds[class1, class_lbl] < 0, 1, 0)
            for class2 in range(class_lbl + 1, self.num_classes):
                ovo_counts[:, class_lbl] += np.where(
                    clf_preds[class_lbl, class2] > 0, 1, 0)
        ovo_argmax = np.argmax(ovo_counts, axis=1)
        refined_cand = self.data.y_train.copy()
        for inst in range(self.num_insts):
            if refined_cand[inst, ovo_argmax[inst]] == 1:
                refined_cand[
                    inst,
                    np.arange(self.num_classes) != ovo_argmax[inst]
                ] = 0

        # Compute one-vs-rest classifiers
        self.ovr_clfs = []
        augmented_x = np.hstack(clf_preds_ordered)
        for class_lbl in range(self.num_classes):
            is_pos = refined_cand[:, class_lbl] == 1
            targets[is_pos] = 1
            targets[~is_pos] = -1

            if np.unique(targets).shape[0] == 2:
                clf = SVC(random_state=self.rng.integers(int(1e6)))
                clf.fit(augmented_x, targets)
                self.ovr_clfs.append(clf)
            else:
                self.ovr_clfs.append(ConstantClassifier())

    def _predict(
        self, data: np.ndarray, candidates: np.ndarray, is_transductive: bool,
    ) -> Optional[SplitResult]:
        if self.ovo_clfs is None or self.ovr_clfs is None:
            return None
        if data.shape[0] == 0:
            return None

        # Compute augmented feature vector
        ovo_results = {
            (class1, class2): clf.decision_function(data)
            for (class1, class2), clf in self.ovo_clfs.items()
        }
        augmented_x = np.hstack([data] + [
            ovo_results[key].reshape(-1, 1)
            for key in sorted(self.ovo_clfs.keys())
        ])
        voting = np.zeros((data.shape[0], self.num_classes), dtype=float)
        for class_lbl in range(self.num_classes):
            # One-vs-one prediction
            for class1 in range(class_lbl):
                voting[:, class_lbl] += np.where(
                    ovo_results[class1, class_lbl] < 0, 1, 0)
            for class2 in range(class_lbl + 1, self.num_classes):
                voting[:, class_lbl] += np.where(
                    ovo_results[class_lbl, class2] > 0, 1, 0)

            # One-vs-rest prediction
            voting[:, class_lbl] += np.where(
                self.ovr_clfs[class_lbl].decision_function(augmented_x) > 0,
                self.mu_param, 0)

        # If in transductive setting, restrict selection to candidate labels
        if is_transductive:
            for i in range(data.shape[0]):
                for j in range(self.num_classes):
                    if candidates[i, j] == 0:
                        voting[i, j] = 0

        # Scale probabilities
        prob_sum = np.sum(voting, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        scaled_probs = voting / prob_sum
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
