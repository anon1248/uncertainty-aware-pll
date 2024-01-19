""" Module for constant classifier. """

import numpy as np


class ConstantClassifier:
    """ Constant classifier. """

    def __init__(self, constant_val: float = 0.0) -> None:
        self.constant_val = constant_val

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """ Returns constant values.

        Args:
            data (np.ndarray): The data specifying the shape.

        Returns:
            np.ndarray: Constant values.
        """

        return self.constant_val * np.ones(data.shape[0])
