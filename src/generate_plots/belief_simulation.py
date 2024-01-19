""" Generate plot "belief_simulation.pdf". """

import random
from fractions import Fraction
from functools import reduce
from itertools import product
from operator import mul
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def yager_combine(
    m_bpas: List[Dict[int, float]], universal_set: int,
) -> Dict[int, float]:
    """ Yager's combination rule. """

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

    return curr_m_bpa


def get_binom_table(max_num):
    """ Create binomial table for fast look-up. """

    dp = [[0] * (max_num + 1) for _ in range(max_num + 1)]

    # Base case: C(i, 0) and C(i, i) are always 1
    for i in range(max_num + 1):
        dp[i][0] = 1
        dp[i][i] = 1

    # Fill in the table using dynamic programming
    for i in range(1, max_num + 1):
        for j in range(1, i + 1):
            dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]

    # Return table
    return dp


binom_table = get_binom_table(1000)


def binom(n, k):
    """ Look-up binomial coefficient. """

    return binom_table[n][k]


def simulation_belief(k_neigh, lbls, p2, p3, rep=100000):
    """ Simulates the belief computation. """

    p4 = 1 - p2 - p3
    vals = list(range(1, (1 << lbls) - 1))
    ps = []
    for v in vals:
        if (v & 1) == 1 and (v & 2) == 2:
            # Case 1: Neighbor's true label and co-occurring label contained
            ps.append(p2 / (2 ** (lbls - 2) - 1))
        elif (v & 1) == 1 and (v & 2) == 0:
            # Case 2: Neighbor's true label contained and co-occurring label not
            ps.append(p3 / (2 ** (lbls - 2)))
        elif (v & 1) == 0:
            # Case 3: Neighbor's true label not contained
            ps.append(p4 / (2 ** (lbls - 1) - 1))

    res = {}
    universe = (1 << lbls) - 1
    for _ in range(rep):
        bpas = [
            {sample: 0.5, universe: 0.5}
            for sample in random.choices(vals, weights=ps, k=k_neigh)
        ]
        combined_bpa = yager_combine(bpas, universe)
        for key, item in combined_bpa.items():
            if key in res:
                res[key] += item
            else:
                res[key] = item

    for k in res:
        res[k] /= rep

    return res.get(1, 0.0)


def expected_belief_counting(
    k: int, lbls: int, h: int, i: int, js: List[int],
) -> int:
    """ Counting part of closed-form expression. """

    # Intersection of k-h sets
    return binom(k, k - h) * ((
        # All combinations of how those k-h sets look
        binom(k - h, i) * reduce(mul, [binom(k - h, j) for j in js], 1)
    ) - sum(
        # Subtract combinations producing at least one full set;
        # candidate sets must not be empty or the universe
        ((-1) ** (a + 1)) * binom(k - h, a) * binom(k - h - a, i - a)
        * reduce(mul, [binom(k - h - a, j - a) for j in js], 1)
        for a in range(1, min(i, *js) + 1)
    ))


def expected_belief_prob(
    k: int, lbls: int, h: int, i: int, p2: Fraction, p3: Fraction,
) -> Fraction:
    """ Probability part of closed-form expression. """

    return (
        # k choices regarding the focal sets of all bpas
        Fraction(1, 2 ** k)
        # Label 1 and 2 contained in i sets
        * ((Fraction(1, 2 ** (lbls - 2) - 1) * p2) ** i)
        # Label 1 and not 2 contained in k-h-i sets
        * ((Fraction(1, 2 ** (lbls - 2)) * p3) ** (k - h - i))
    )


def expected_belief(
    k: int, lbls: int, p2: Fraction, p3: Fraction,
) -> Fraction:
    """ Closed-form expression for belief. """

    if lbls <= 2 or k <= 0:
        raise ValueError()
    return sum(
        expected_belief_counting(k, lbls, h, i, js)
        * expected_belief_prob(k, lbls, h, i, p2, p3)
        for h in range(k)
        for i in range(k - h)
        for js in product(range(k - h), repeat=lbls - 2)
    )


def simulation_belief_coocc(k_neigh, lbls, p2, p3, rep=100000):
    """ Simulation of belief in cooccurring label. """

    p4 = 1 - p2 - p3
    vals = list(range(1, (1 << lbls) - 1))
    ps = []
    for v in vals:
        if (v & 1) == 1 and (v & 2) == 2:
            # Case 1: Neighbor's true label and co-occurring label contained
            ps.append(p2 / (2 ** (lbls - 2) - 1))
        elif (v & 1) == 1 and (v & 2) == 0:
            # Case 2: Neighbor's true label contained and co-occurring label not
            ps.append(p3 / (2 ** (lbls - 2)))
        elif (v & 1) == 0:
            # Case 3: Neighbor's true label not contained
            ps.append(p4 / (2 ** (lbls - 1) - 1))

    res = {}
    universe = (1 << lbls) - 1
    for _ in range(rep):
        bpas = [
            {sample: 0.5, universe: 0.5}
            for sample in random.choices(vals, weights=ps, k=k_neigh)
        ]
        combined_bpa = yager_combine(bpas, universe)
        for key, item in combined_bpa.items():
            if key in res:
                res[key] += item
            else:
                res[key] = item

    for k in res.keys():
        res[k] /= rep

    return res.get(2, 0.0)


def expected_belief_coocc_counting(
    k: int, lbls: int, h: int, t: int, js: List[int],
) -> int:
    """ Counting part of closed-form expression. """

    # Intersection of k-h sets
    return binom(k, k - h) * ((
        # All combinations of how those k-h sets look
        binom(k - h, t) * reduce(mul, [binom(k - h, j) for j in js], 1)
    ) - sum(
        # Subtract combinations producing at least one full set;
        # candidate sets must not be empty or the universe
        ((-1) ** (a + 1)) * binom(k - h, a) * binom(k - h - a, t - a)
        * reduce(mul, [binom(k - h - a, j - a) for j in js], 1)
        for a in range(1, min(t, *js) + 1)
    ))


def expected_belief_coocc_prob(
    k: int, lbls: int, h: int, t: int, p2: Fraction, p3: Fraction,
) -> Fraction:
    """ Probability part of closed-form expression. """

    return (
        # k choices regarding the focal sets of all bpas
        Fraction(1, 2 ** k)
        # Label 1 and 2 contained in t sets
        * ((Fraction(1, 2 ** (lbls - 2) - 1) * p2) ** t)
        # Not label 1 but 2 contained in k-h-t sets
        * ((Fraction(1, 2 ** (lbls - 1) - 1) * (1 - p2 - p3)) ** (k - h - t))
    )


def expected_belief_coocc(
    k: int, lbls: int, p2: Fraction, p3: Fraction,
) -> Fraction:
    """ Closed-form expression for belief in cooccurring label. """

    if lbls <= 2 or k <= 0:
        raise ValueError()
    return sum(
        expected_belief_coocc_counting(k, lbls, h, t, js)
        * expected_belief_coocc_prob(k, lbls, h, t, p2, p3)
        for h in range(k)
        for t in range(k - h)
        for js in product(range(k - h), repeat=lbls - 2)
    )


config = (3, 0.4, 0.35)
res = [(k, expected_belief(k, *config), simulation_belief(k, *config))
       for k in tqdm(range(1, 101))]
res_coocc = [(
    k, expected_belief_coocc(k, *config),
    simulation_belief_coocc(k, *config)
) for k in tqdm(range(1, 101))]

npres = np.array(res)
npres_coocc = np.array(res_coocc)

plt.style.use("tableau-colorblind10")

fs = 9
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath, amssymb}"
plt.rc("font", family="serif")
plt.rc("savefig", bbox="tight")

plt.rc("xtick", labelsize=fs)
plt.rc("ytick", labelsize=fs)

plt.figure(figsize=(6, 2.5))

plt.plot(npres[:, 0], npres[:, 1], "-", color="C1",
         label="$\\mathbb{E}_{\\mathbb{P}}\\!\\left[ \\operatorname{bel}^{\\left(\\operatorname{\\tilde{m}}\\right)}\\!\\left( \\left\\lbrace \\tilde{y} \\right\\rbrace \\right) \\mid X = \\tilde{x} \\right]$ -- Calculation")
plt.plot(npres[:, 0], npres[:, 2], "--", color="C2",
         label="$\\mathbb{E}_{\\mathbb{P}}\\!\\left[ \\operatorname{bel}^{\\left(\\operatorname{\\tilde{m}}\\right)}\\!\\left( \\left\\lbrace \\tilde{y} \\right\\rbrace \\right) \\mid X = \\tilde{x} \\right]$ -- Simulation")

plt.plot(npres_coocc[:, 0], npres_coocc[:, 1], "-.", color="C3",
         label="$\\mathbb{E}_{\\mathbb{P}}\\!\\left[ \\operatorname{bel}^{\\left(\\operatorname{\\tilde{m}}\\right)}( \\lbrace \\tilde{y}^{\\left(\\tilde{y}\\right)}_{\\operatorname{c}} \\rbrace ) \\mid X = \\tilde{x} \\right]$ -- Calculation")
plt.plot(npres_coocc[:, 0], npres_coocc[:, 2], ":", color="C4",
         label="$\\mathbb{E}_{\\mathbb{P}}\\!\\left[ \\operatorname{bel}^{\\left(\\operatorname{\\tilde{m}}\\right)}( \\lbrace \\tilde{y}^{\\left(\\tilde{y}\\right)}_{\\operatorname{c}} \\rbrace ) \\mid X = \\tilde{x} \\right]$ -- Simulation")

plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 0.4)

plt.xlabel("Number of neighbors $k$", fontsize=fs)
plt.ylabel("Belief", fontsize=fs)

plt.savefig("paper/plots/belief-simulation.pdf")
