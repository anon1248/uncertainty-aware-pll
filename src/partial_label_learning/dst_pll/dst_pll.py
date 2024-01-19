""" Module for DST-PLL. """

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from partial_label_learning.data import Datasplit
from partial_label_learning.dst_pll.dst_pll_helper import (
    CandidateLabelsEncoder, yager_combine)
from partial_label_learning.result import SplitResult


class DstPll:
    """
    Dempster-Shafer Theory for Partial Label Learning.
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        k_neighbors: int = 10,
    ) -> None:
        """ Inits the DST-PLL algorithm.

        Args:
            data (Datasplit): The dataset.
            rng (np.random.Generator): A random generator.
            k_neighbors (int, optional): The number of neighbors.
            Defaults to 10.
        """

        self.data = data
        self.rng = rng
        self.k_neighbors = k_neighbors
        self.num_classes = self.data.orig_dataset.l_classes

        # Compute nearest neighbors
        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=1)
        self.knn.fit(self.data.x_train)

        # Label encoder
        self.label_encoder = CandidateLabelsEncoder(self.num_classes)

    def get_single_item_pred(
        self, m_bpa: Dict[int, float],
    ) -> Tuple[List[int], bool]:
        """ Compute maximum beliefs and whether we are confident.

        Args:
            m_bpa (Dict[int, float]): The bpa.

        Returns:
            Tuple[List[int], bool]: Indices of maximum belief
            and whether we are confident.
        """

        # Find maximum single-item belief
        curr_belief_argmax_list = []
        curr_belief_max = -1.0
        curr_class_idx = 1
        for class_lbl in range(self.num_classes):
            belief_of_class = m_bpa.get(curr_class_idx, 0.0)
            if belief_of_class > 0:
                if belief_of_class > curr_belief_max:
                    curr_belief_max = belief_of_class
                    curr_belief_argmax_list = [class_lbl]
                elif belief_of_class == curr_belief_max:
                    curr_belief_argmax_list.append(class_lbl)
            curr_class_idx <<= 1

        # Whether items are more plausible than max belief
        still_plausible_num = 0
        curr_class_idx = 1
        for _ in range(self.num_classes):
            plausibility = sum(
                subset_val for subset, subset_val in m_bpa.items()
                if (subset & curr_class_idx) == curr_class_idx
            )
            if plausibility >= curr_belief_max:
                still_plausible_num += 1
            curr_class_idx <<= 1

        return curr_belief_argmax_list, still_plausible_num == 1

    def infer_labeling(
        self,
        nn_indices: np.ndarray,
        is_train: bool,
    ) -> SplitResult:
        """ Infer labeling by combining evidence from neighbors.

        Args:
            partial_targets (np.ndarray): The partial targets.
            nn_indices (np.ndarray): The neighbor indices.
            is_train (bool): Whether in training mode.

        Returns:
            SplitResult: Prediction results and more.
        """

        # Encode candidate lists as bit strings;
        # Python integers have arbitrary bit length
        num_classes_mask = (1 << self.num_classes) - 1
        train_targets_enc = [
            self.label_encoder.encode_candidate_list(y_row)
            for y_row in self.data.y_train
        ]

        # Combine evidence from nearest neighbors
        # using Dempster's rule of combination
        pred_list: List[int] = []
        pred_with_unc_list: List[int] = []
        guessing: List[bool] = []
        for inst, train_inst_neighbors in enumerate(nn_indices):
            # We are sure that the answer is from the given candidate set
            inst_candidates = (
                train_targets_enc[inst]
                if is_train else num_classes_mask
            )
            evidence_to_combine = [{inst_candidates: 1.0}]

            # Combine evidence from neighbors using Yager's rule
            for train_neighbor_idx in map(int, train_inst_neighbors):
                # Retrieve candidates of neighbor
                neighbor_candidates = train_targets_enc[train_neighbor_idx]

                # If all probability mass is already allotted
                # to candidates in evidence or evidence is disjoint,
                # do not use it since it has no influence
                if inst_candidates & neighbor_candidates \
                        in (0, inst_candidates):
                    continue

                # Append evidence
                evidence_to_combine.append({
                    (neighbor_candidates & inst_candidates): 0.5,
                    inst_candidates: 0.5,
                })

            # Combine evidence
            m_bpa = yager_combine(evidence_to_combine, inst_candidates)

            # Extract prediction
            single_pred, is_confident = self.get_single_item_pred(m_bpa)

            # Determine most-likely subset
            if single_pred:
                # Take single element that is most likely
                most_likely_subset = single_pred
            else:
                # Else, determine subset with highest belief
                # and smallest cardinality when tied
                most_likely_subset_encoded = sorted(
                    list(m_bpa.items()),
                    key=lambda t: (-t[1], t[0].bit_count()),
                )[0][0]
                most_likely_subset = sorted(list(map(int, np.where(np.array(
                    self.label_encoder.decode_candidate_list(
                        most_likely_subset_encoded)) == 1)[0])))

            # Randomly pick if most likely subset contains more than one item
            if len(most_likely_subset) != 1:
                choosen_elem = int(self.rng.choice(most_likely_subset))
                is_guessing = True
            else:
                choosen_elem = most_likely_subset[0]
                is_guessing = False

            # Save predictions
            pred_list.append(choosen_elem)
            guessing.append(is_guessing)
            pred_with_unc_list.append(
                choosen_elem
                if is_confident and not is_guessing
                else self.num_classes
            )

        # Return predictions
        return SplitResult(
            pred=np.array(pred_list),
            is_sure_pred=np.array(pred_with_unc_list) != self.num_classes,
            is_guessing=np.array(guessing),
        )

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        # Compute nearest neighbors for each instance
        nn_indices = self.knn.kneighbors(return_distance=False)

        # Return train predictions
        return self.infer_labeling(nn_indices, True)

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        # Return if nothing to predict
        if self.data.x_test.shape[0] == 0:
            return None

        # Compute nearest neighbors for each instance
        nn_indices = self.knn.kneighbors(
            self.data.x_test, return_distance=False)

        # Return test predictions
        return self.infer_labeling(nn_indices, False)
