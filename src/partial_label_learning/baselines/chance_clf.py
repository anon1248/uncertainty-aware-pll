""" Module for chance classifier. """

from typing import Optional

import numpy as np

from partial_label_learning.data import Datasplit
from partial_label_learning.result import SplitResult


class ChanceClf:
    """ Simple guessing classifier. """

    def __init__(
        self, data: Datasplit, rng: np.random.Generator,
    ) -> None:
        self.data = data
        self.rng = rng

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        y_train_pred_list = []
        for y_row in self.data.y_train:
            y_train_pred_list.append(self.rng.choice(np.where(y_row != 0)[0]))
        return SplitResult(
            pred=np.array(y_train_pred_list),
            is_sure_pred=np.zeros(len(y_train_pred_list), dtype=bool),
            is_guessing=np.ones(len(y_train_pred_list), dtype=bool),
        )

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        if self.data.x_test.shape[0] == 0:
            return None

        y_test_pred_list = []
        for y_row in self.data.y_test:
            y_test_pred_list.append(self.rng.choice(y_row.shape[0]))
        return SplitResult(
            pred=np.array(y_test_pred_list),
            is_sure_pred=np.zeros(len(y_test_pred_list), dtype=bool),
            is_guessing=np.ones(len(y_test_pred_list), dtype=bool),
        )
