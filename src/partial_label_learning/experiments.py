""" Main experiment module. """

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.svm import SVC

from partial_label_learning.baselines.chance_clf import ChanceClf
from partial_label_learning.baselines.fully_supervised_clf import \
    FullySupervisedClf
from partial_label_learning.baselines.ovr_clf import OvrPll
from partial_label_learning.data import Dataset, Datasplit
from partial_label_learning.dst_pll.dst_pll import DstPll
from partial_label_learning.relatedwork.clpl_2011 import Clpl
from partial_label_learning.relatedwork.ipal_2015 import Ipal
from partial_label_learning.relatedwork.lsb_cmm_2012 import LsbCmm
from partial_label_learning.relatedwork.m3pl_2016 import M3Pl
from partial_label_learning.relatedwork.paloc_2018 import Paloc
from partial_label_learning.relatedwork.pl_ecoc_2017 import PlEcoc
from partial_label_learning.relatedwork.pl_knn_2005 import PlKnn
from partial_label_learning.relatedwork.pl_svm_2008 import PlSvm
from partial_label_learning.relatedwork.sure_2019 import Sure
from partial_label_learning.result import Result

COLUMNS = [
    "dataset", "algo", "train_frac", "r_candidates",
    "p_partially_labeled", "eps_coocc", "seed",
    "train_acc", "test_acc", "train_mcc", "test_mcc",
    "train_frac_guessing", "test_frac_guessing",
    "train_frac_sure", "test_frac_sure",
    "train_acc_sure", "test_acc_sure",
    "train_mcc_sure", "test_mcc_sure",
    "runtime",
]


def get_eval_tuple(
    dataset_name: str, algo_name: str, train_frac: float,
    r_candidates: int, p_partially_labeled: float,
    eps_coocc: float, seed: int,
    data: Datasplit, result: Result,
    runtime: float, debug: bool,
) -> List:
    """ Get a single result row.

    Args:
        dataset_name (str): The dataset name.
        algo_name (str): The algo name.
        train_frac (float): The train-set fraction.
        r_candidates (int): The number of additional candidates.
        p_partially_labeled (float): Percent of data partially labeled.
        eps_coocc (float): The parameter of the geometric distribution.
        seed (int): The seed.
        data (Datasplit): The data used.
        result (Result): The prediction results.
        runtime (float): Runtime in seconds.
        debug (bool): Whether to print results.

    Returns:
        List: The result row.
    """

    # Train set evaluation
    train_acc = accuracy_score(
        data.y_true_train, result.train_result.pred
    )
    train_mcc = matthews_corrcoef(
        data.y_true_train, result.train_result.pred
    )
    train_frac_guessing = result.train_result.frac_guessing()
    train_frac_sure = result.train_result.frac_sure_predictions()
    train_acc_sure = accuracy_score(
        data.y_true_train[result.train_result.is_sure_pred],
        result.train_result.sure_predictions(),
    ) if train_frac_sure > 0 else 0.0
    train_mcc_sure = matthews_corrcoef(
        data.y_true_train[result.train_result.is_sure_pred],
        result.train_result.sure_predictions(),
    ) if train_frac_sure > 0 else 0.0

    # Test set evaluation
    if data.y_true_test.shape[0] != 0:
        test_acc = accuracy_score(
            data.y_true_test, result.get_test_result().pred
        )
        test_mcc = matthews_corrcoef(
            data.y_true_test, result.get_test_result().pred
        )
        test_frac_guessing = result.get_test_result().frac_guessing()
        test_frac_sure = result.get_test_result().frac_sure_predictions()
        test_acc_sure = accuracy_score(
            data.y_true_test[result.get_test_result().is_sure_pred],
            result.get_test_result().sure_predictions(),
        ) if test_frac_sure > 0 else 0.0
        test_mcc_sure = matthews_corrcoef(
            data.y_true_test[result.get_test_result().is_sure_pred],
            result.get_test_result().sure_predictions(),
        ) if test_frac_sure > 0 else 0.0
    else:
        test_acc, test_mcc = 0.0, 0.0
        test_frac_guessing, test_frac_sure = 0.0, 0.0
        test_acc_sure, test_mcc_sure = 0.0, 0.0

    # Build result list
    res_tup = [
        # Dataset
        dataset_name,
        # Algorithm
        algo_name, f"{train_frac:.1f}",
        # Dataset configuration
        f"{r_candidates}", f"{p_partially_labeled:.1f}", f"{eps_coocc:.1f}",
        # Seed
        f"{seed}",
        # Accuracy
        f"{train_acc:.6f}", f"{test_acc:.6f}",
        # MCC
        f"{train_mcc:.6f}", f"{test_mcc:.6f}",
        # Fraction guessing
        f"{train_frac_guessing:.6f}", f"{test_frac_guessing:.6f}",
        # Fraction sure predictions
        f"{train_frac_sure:.6f}", f"{test_frac_sure:.6f}",
        # Accuracy sure predictions
        f"{train_acc_sure:.6f}", f"{test_acc_sure:.6f}",
        # MCC sure predictions
        f"{train_mcc_sure:.6f}", f"{test_mcc_sure:.6f}",
        # Runtime
        f"{runtime:.6f}",
    ]
    if debug:
        res_tup[0] = f"{res_tup[0]: >27}"
        res_tup[1] = f"{res_tup[1]: >15}"
        res_tup[2] = f"{res_tup[2]: >3}"
        res_tup[3] = f"{res_tup[3]: >1}"
        res_tup[4] = f"{res_tup[4]: >3}"
        res_tup[5] = f"{res_tup[5]: >3}"
        res_tup[6] = f"{res_tup[6]: >2}"
        print(", ".join(res_tup))
    return res_tup


def reset_rng(seed: int) -> np.random.Generator:
    """ Creates a new random engine.

    Args:
        seed (int): The seed to use.

    Returns:
        np.random.Generator: The random generator.
    """

    return np.random.Generator(np.random.PCG64(seed))


def run_experiment(
    # Dataset
    dataset_name: str, dataset: Dataset, train_frac: float,
    # Properties of augmented dataset
    r_candidates: int, p_partially_labeled: float, eps_coocc: float,
    # Misc
    seed: int, debug: bool, augment: bool, algo: str,
):
    """ Runs a single experiment. """

    # Reset random engine; assign all algorithms different seeds
    # to ensure all results are mutually independent
    seed_offset = 0
    rng = reset_rng(seed + seed_offset)

    # Create datasplit
    if augment:
        aug_dataset = dataset.augment_targets(
            rng, r_candidates, p_partially_labeled, eps_coocc)
    else:
        aug_dataset = dataset
    datasplit = Datasplit.create_random_split_from_dataset(
        aug_dataset, seed, rng, test_size=1.0 - train_frac)

    # Run experiments
    res = []
    seed_offset += 1
    if algo in ("all", "baselines", "chance"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # Naive chance baseline
        start = time.process_time()
        chance_clf = ChanceClf(datasplit.copy(), rng)
        result = Result(
            train_result=chance_clf.get_train_pred(),
            test_result=chance_clf.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "chance", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("baselines", "ovr-svm"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # One-vs-rest support-vector machine
        start = time.process_time()
        ovr_svm = OvrPll(
            datasplit.copy(), rng, SVC(
                random_state=rng.integers(int(1e6)),
            ),
        )
        result = Result(
            train_result=ovr_svm.get_train_pred(),
            test_result=ovr_svm.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "ovr-svm", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("baselines", "supervised-svm"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # Fully supervised support-vector machine
        start = time.process_time()
        upper_bound_svm = FullySupervisedClf(
            datasplit.copy(), rng, SVC(
                random_state=rng.integers(int(1e6)),
            ),
        )
        result = Result(
            train_result=upper_bound_svm.get_train_pred(),
            test_result=upper_bound_svm.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "supervised-svm", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "pl-knn-2005"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # PL-KNN
        start = time.process_time()
        knn = PlKnn(datasplit.copy(), rng)
        result = Result(
            train_result=knn.get_train_pred(),
            test_result=knn.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "pl-knn-2005", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "pl-svm-2008"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # PL-SVM
        start = time.process_time()
        pl_svm = PlSvm(datasplit.copy(), rng)
        pl_svm.fit()
        result = Result(
            train_result=pl_svm.get_train_pred(),
            test_result=pl_svm.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "pl-svm-2008", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "clpl-2011"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # CLPL
        start = time.process_time()
        clpl = Clpl(datasplit.copy(), rng)
        clpl.fit()
        result = Result(
            train_result=clpl.get_train_pred(),
            test_result=clpl.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "clpl-2011", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "lsb-cmm-2012"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # LSB-CMM
        start = time.process_time()
        lsb_cmm = LsbCmm(datasplit.copy(), rng)
        lsb_cmm.fit()
        result = Result(
            train_result=lsb_cmm.get_train_pred(),
            test_result=lsb_cmm.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "lsb-cmm-2012", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "ipal-2015"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # IPAL
        start = time.process_time()
        ipal = Ipal(datasplit.copy(), rng)
        ipal.fit()
        result = Result(
            train_result=ipal.get_train_pred(),
            test_result=ipal.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "ipal-2015", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "m3pl-2016"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # M3PL
        start = time.process_time()
        m3pl = M3Pl(datasplit.copy(), rng)
        m3pl.fit()
        result = Result(
            train_result=m3pl.get_train_pred(),
            test_result=m3pl.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "m3pl-2016", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("competitors", "pl-ecoc-2017"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # PL-ECOC
        start = time.process_time()
        pl_ecoc = PlEcoc(datasplit.copy(), rng)
        pl_ecoc.fit()
        result = Result(
            train_result=pl_ecoc.get_train_pred(),
            test_result=pl_ecoc.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "pl-ecoc-2017", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "paloc-2018"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # PALOC
        start = time.process_time()
        paloc = Paloc(datasplit.copy(), rng)
        paloc.fit()
        result = Result(
            train_result=paloc.get_train_pred(),
            test_result=paloc.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "paloc-2018", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "sure-2019"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # SURE
        start = time.process_time()
        sure = Sure(datasplit.copy(), rng)
        sure.fit()
        result = Result(
            train_result=sure.get_train_pred(),
            test_result=sure.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "sure-2019", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    seed_offset += 1
    if algo in ("all", "competitors", "dst-pll"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # DST-PLL
        start = time.process_time()
        dst_pll = DstPll(datasplit.copy(), rng)
        result = Result(
            train_result=dst_pll.get_train_pred(),
            test_result=dst_pll.get_test_pred(),
        )
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            dataset_name, "dst-pll", train_frac,
            r_candidates, p_partially_labeled, eps_coocc,
            seed, datasplit, result, time_used, debug,
        ))

    return res


def print_datasets(datasets: Dict[str, Dataset], prefix: str = "") -> None:
    """ Print dataset characteristics.

    Args:
        datasets (Dict[str, Dataset]): The datasets.
    """

    # Print dataset stats
    rows: List[Tuple] = []
    for i, (dataset_name, dataset) in enumerate(sorted(datasets.items())):
        rows.append((
            i, dataset_name, dataset.n_samples, dataset.m_features, dataset.l_classes
        ))
    dataset_df = pd.DataFrame.from_records(rows, columns=[
        "idx", "dataset", "n_samples", "m_features", "l_classes",
    ], index="idx")
    dataset_df.to_csv(f"results/{prefix}datasets_used.csv")
    print()
    print(dataset_df)
    print()
