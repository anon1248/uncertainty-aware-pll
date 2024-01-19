""" Helper methods for DST-PLL. """

from typing import Dict, List, Tuple, Union

import numpy as np


def yager_combine(
    m_bpas: List[Dict[int, float]],
    universal_set: int,
    prune_prob: float = 1e-10,
) -> Dict[int, float]:
    """ Yager rule of combination.

    Args:
        m_bpas (List[Dict[int, float]]): The evidences to combine.
        universal_set (int): The universal set in this context.
        prune_prob (float): Prune probabilities smaller than this.
        Defaults to 1e-10.

    Returns:
        Dict[int, float]: The combined evidence.
    """

    # Combine observations
    curr_m_bpa = {universal_set: 1.0}
    for next_m_bpa in m_bpas:
        new_m_bpa = {}
        for set1, set1_prob in curr_m_bpa.items():
            for set2, set2_prob in next_m_bpa.items():
                intersect = set1 & set2
                prob_prod = set1_prob * set2_prob
                # Increase probability of intersecting evidence
                if intersect not in new_m_bpa:
                    new_m_bpa[intersect] = prob_prod
                else:
                    new_m_bpa[intersect] += prob_prod
        curr_m_bpa = new_m_bpa

    # Assign probability of empty set to universal set
    if 0 in curr_m_bpa:
        if universal_set not in curr_m_bpa:
            curr_m_bpa[universal_set] = curr_m_bpa[0]
        else:
            curr_m_bpa[universal_set] += curr_m_bpa[0]
        del curr_m_bpa[0]

    # Prune too small probabilities
    for subset in list(curr_m_bpa.keys()):
        if curr_m_bpa[subset] < prune_prob:
            del curr_m_bpa[subset]

    # Rescale
    sum_probs = sum(curr_m_bpa.values())
    if sum_probs != 1.0:
        for subset in curr_m_bpa:
            curr_m_bpa[subset] /= sum_probs

    return curr_m_bpa


def get_total_uncertainty(
    m_bpa: Dict[int, float], universe: int,
) -> float:
    """ Returns the total uncertainty of a bpa.

    Args:
        m_bpa (Dict[int, float]): The bpa.
        curr_universe (int): The universe.

    Returns:
        float: The uncertainty in [0, 1].
    """

    # No uncertainty if label already known
    if universe.bit_count() == 1:
        return 0.0

    # Compute belief from bpa
    curr_belief = {
        k: 0.0 for k in m_bpa
    }
    for subset1 in m_bpa:
        for subset2, subset_val in m_bpa.items():
            if (subset1 & subset2) == subset2:
                curr_belief[subset1] += subset_val
    p_x: Dict[int, float] = {}
    curr_universe = universe

    # Iteratively find maximum belief ratio sets
    while curr_belief and curr_universe:
        # Find maximal belief per set size
        max_belief_ratio, _, max_belief, max_val_subset = max(
            (
                belief_val / subset.bit_count(),
                subset.bit_count(),
                belief_val,
                subset,
            )
            for subset, belief_val in curr_belief.items()
            if (subset & curr_universe) == subset
        )
        curr_universe &= ~max_val_subset

        # Set probability for members of max-belief-ratio set
        class_lbl_enc = 1
        for class_lbl in range(universe.bit_length()):
            if (class_lbl_enc & max_val_subset) == class_lbl_enc:
                p_x[class_lbl] = max_belief_ratio
            class_lbl_enc <<= 1

        # Update belief frame
        new_belief: Dict[int, float] = {}
        for subset, subset_val in curr_belief.items():
            if (subset & max_val_subset) == max_val_subset and \
                    subset != max_val_subset:
                new_belief[subset ^ max_val_subset] = \
                    subset_val - max_belief
        curr_belief = new_belief

    # Return total uncertainty
    p_x_arr = np.array(list(p_x.values()))
    res = float(
        -np.sum(p_x_arr * np.log2(p_x_arr))
        / np.log2(universe.bit_count())
    )
    return min(1.0, max(0.0, res))


def get_disaggregated_uncertainty(
    m_bpa: Dict[int, float], universe: int,
) -> Tuple[float, float]:
    """ Returns the aleatoric and epistemic uncertainty.

    Args:
        m_bpa (Dict[int, float]): The bpa.
        universe (int): The universe.

    Returns:
        Tuple[float, float]: Aleatoric and epistemic uncertainty.
    """

    # Get total uncertainy
    total_unc = get_total_uncertainty(m_bpa, universe)

    # Compute hartley measure
    epistemic_unc = float(sum(
        v * np.log2(k.bit_count())
        for k, v in m_bpa.items()
    ) / np.log2(universe.bit_count())) \
        if universe.bit_count() != 1 else 0.0

    # Check bounds of uncertainties
    if not (
        0 <= total_unc - epistemic_unc <= 1 and
        0 <= epistemic_unc <= 1 and
        0 <= total_unc <= 1
    ):
        raise ValueError("Bug in uncertainty disaggregation.")

    # Return aleatoric and epistemic uncertainty
    return total_unc - epistemic_unc, epistemic_unc


class CandidateLabelsEncoder:
    """ Class for encoding candidate label sets. """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def encode_candidate_list(
        self, candidates: Union[np.ndarray, List[int]],
    ) -> int:
        """ Encodes a candidate list into an integer.

        Args:
            candidates (Union[np.ndarray, List[int]]): The binary candidate vector.

        Returns:
            int: The encoded candidate list.
        """

        return sum(
            (1 << code_pos) * int(candidates[code_pos])
            for code_pos in range(self.num_classes)
        )

    def decode_candidate_list(self, encoded_candidates: int) -> List[int]:
        """ Decodes an encoded candidate list back to a list.

        Args:
            encoded_candidates (int): The encoded representation.

        Returns:
            List[int]: The binary class list.
        """

        return [
            1
            if ((1 << class_lbl) & encoded_candidates) != 0
            else 0
            for class_lbl in range(self.num_classes)
        ]
