# Existing evaluation methods
import numpy as np
from itertools import combinations
from collections import Counter


from itertools import combinations
from collections import Counter
import numpy as np

def stability_analysis(feature_sets, N, k):
    """
    feature_sets: list of feature sets from repeated experiments
    N: total number of features
    k: number of features selected each time
    """
    sets = [set(fs) for fs in feature_sets]

    # Selection frequency
    freq = Counter(f for s in sets for f in s)
    selection_freq = {f: freq[f] / len(sets) for f in range(N)}

    # Jaccard
    jaccards = []
    kunchevas = []
    for a, b in combinations(sets, 2):
        inter = len(a & b)
        union = len(a | b)
        if union > 0:
            jaccards.append(inter / union)
        # Kuncheva
        kunchevas.append((inter * N - k**2) / (k * (N - k)))

    mean_jaccard = np.mean(jaccards) if jaccards else np.nan
    mean_kuncheva = np.mean(kunchevas) if kunchevas else np.nan

    return {
        "mean_jaccard": mean_jaccard,
        "mean_kuncheva": mean_kuncheva,
        "selection_freq": selection_freq
    }


from itertools import combinations
from collections import Counter
import numpy as np

def stability_used_feature(feature_sets):
    sets = [set(fs) for fs in feature_sets]

    # Jaccard
    jaccards = []
    kunchevas = []
    for a, b in combinations(sets, 2):
        inter = len(a & b)
        union = len(a | b)
        if union > 0:
            jaccards.append(inter / union)

    mean_jaccard = np.mean(jaccards) if jaccards else np.nan

    return round(np.mean(mean_jaccard), 4)

def eval_against_truth(selected, true_set):
    """
    Evaluate how well the selected features match the true features.

    Parameters:
    ----------
    selected : array-like
        Selected feature indices.
    true_set : array-like
        True feature indices.

    Returns:
    -------
    precision : float
        Precision.
    recall : float
        Recall.
    f1 : float
        F1 score.
    """
    try:
        selected = set(selected)
        true_set = set(true_set)

        tp = len(selected & true_set)
        precision = tp / len(selected) if selected else 0
        recall = tp / len(true_set) if true_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1
    except Exception as e:
        print(f"eval_against_truth错误: {e}")
        return 0.0, 0.0, 0.0


def ev_truth(f_set, N, true_set):
    """
    Evaluate a feature-selection method against the true features.

    Parameters:
    ----------
    f_set : list of arrays
        List of feature-selection results.
    N : int
        Number of evaluation runs.
    true_set : array-like
        True feature indices.

    Returns:
    -------
    mean_precision : float
        Mean precision.
    mean_recall : float
        Mean recall.
    mean_f1 : float
        Mean F1 score.
    """
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(min(N, len(f_set))):  # Ensure the index stays within f_set
        try:
            precision, recall, f1 = eval_against_truth(f_set[i], true_set)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        except Exception as e:
            print(f"第 {i} 次评估失败: {e}")
            precisions.append(0.0)
            recalls.append(0.0)
            f1_scores.append(0.0)

    # Compute averages; return 0 if a list is empty
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    mean_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return mean_precision, mean_recall, mean_f1


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def multiple_run_consistency(importance_runs, feature_names=None):
    n_runs = len(importance_runs)
    n_features = len(importance_runs[0])

    if feature_names is None:
        feature_names = [f'Feature_{i + 1}' for i in range(n_features)]

    rankings = []
    for run in importance_runs:
        # Convert to rankings, with rank 1 as most important
        ranks = np.argsort(run)[::-1]
        rankings.append(ranks)

    rankings = np.array(rankings)

    # 2. Compute the average ranking
    avg_ranks = np.mean(rankings, axis=0)

    # 3. Compute pairwise correlations
    spearman_matrix = np.zeros((n_runs, n_runs))
    kendall_matrix = np.zeros((n_runs, n_runs))

    for i in range(n_runs):
        for j in range(i, n_runs):
            if i == j:
                spearman_matrix[i, j] = 1.0
                kendall_matrix[i, j] = 1.0
            else:
                spearman_corr, _ = spearmanr(rankings[i], rankings[j])
                kendall_corr, _ = kendalltau(rankings[i], rankings[j])

                spearman_matrix[i, j] = spearman_corr
                spearman_matrix[j, i] = spearman_corr
                kendall_matrix[i, j] = kendall_corr
                kendall_matrix[j, i] = kendall_corr

    # 4. Compute overall consistency metrics
    # Extract the upper triangle, excluding the diagonal
    spearman_values = spearman_matrix[np.triu_indices(n_runs, k=1)]
    kendall_values = kendall_matrix[np.triu_indices(n_runs, k=1)]

    mean_spearman = round(np.mean(spearman_values), 4)
    mean_kendall = round(np.mean(kendall_values), 4)
    #return mean_spearman, mean_kendall
    return mean_spearman

def slope_trend_similarity(shapes, xs):
    slopes = np.diff(shapes, axis=1) / np.diff(xs)
    # cosine of slopes
    from sklearn.metrics.pairwise import cosine_similarity
    #cos_mat = cosine_similarity(slopes - slopes.mean(axis=1, keepdims=True))
    cos_mat = cosine_similarity(slopes)
    return np.mean(cos_mat[np.triu_indices_from(cos_mat, k=1)])



from scipy import stats

def make_baseline(X_train, continuous_idx, categorical_idx):
    """
    X_train: shape (n, d)
    continuous_idx: list of continuous feature indices
    categorical_idx: list of categorical feature indices
    """
    baseline = np.zeros(X_train.shape[1], dtype=X_train.dtype)

    # Continuous variables -> mean
    for j in continuous_idx:
        baseline[j] = np.mean(X_train[:, j].astype(float))

    # Categorical variables -> mode
    for j in categorical_idx:
        mode_result = stats.mode(X_train[:, j], keepdims=False)
        baseline[j] = mode_result.mode

    return baseline


