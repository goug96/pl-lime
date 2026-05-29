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

def deletion_auc(model, x, feature_ranking, baseline, class_id=0):
    """
    x: shape (1, d)
    baseline: shape (d,)
    """
    d = x.shape[1]
    scores = []

    x_del = x.copy()
    scores.append(model.predict_proba(x_del)[0, class_id])

    for j in feature_ranking:
        x_del[:, j] = baseline[j]
        scores.append(model.predict_proba(x_del)[0, class_id])

    auc = np.trapz(scores, dx=1/d)
    return auc

import numpy as np

def deletion_auc1(model, x, feature_ranking, baseline, class_id=0):
    """
    model: classifier with predict_proba
    x: array of shape (1, d)
    feature_ranking: list/array of feature indices, ordered from most important to least important
    baseline: array of shape (d,)
    """
    d = x.shape[1]
    x_del = x.copy()
    scores = [model.predict_proba(x_del)[0, class_id]]

    for j in feature_ranking:
        x_del[:, j] = baseline[j]
        scores.append(model.predict_proba(x_del)[0, class_id])

    scores = np.array(scores)

    # deletion ratio on x-axis
    xs = np.linspace(0, len(feature_ranking) / d, len(feature_ranking) + 1)

    auc = np.trapz(scores, xs)
    return auc, xs, scores


def deletion_auc_cat(model, x, feature_ranking, baseline, class_id=0):
    """
    x: shape (1, d)
    baseline: shape (d,)
    """
    d = x.shape[1]
    scores = []

    x_del = x.copy()
    scores.append(model(x_del)[0, class_id])

    for j in feature_ranking:
        x_del[:, j] = baseline[j]
        scores.append(model(x_del)[0, class_id])

    auc = np.trapz(scores, dx=1/d)
    return auc

import numpy as np

def deletion_auc_cat1(predict_fn, x, feature_ranking, baseline, class_id=0):
    """
    x: shape (1, d)
    baseline: shape (d,)
    feature_ranking: ordered from most important to least important
    """
    d = x.shape[1]
    x_del = x.copy()
    scores = [predict_fn(x_del)[0, class_id]]

    for j in feature_ranking:
        x_del[:, j] = baseline[j]
        scores.append(predict_fn(x_del)[0, class_id])


    xs = np.linspace(0, len(feature_ranking) / d, len(feature_ranking) + 1)
    auc = np.trapz(scores, xs)
    return auc, xs, np.array(scores)


def deletion_auc_r(predict_fn, x, feature_ranking, baseline,class_id=0):
    """
    Deletion AUC calculation for regression.
    x: shape (1, n_features)
    baseline: shape (n_features,)
    feature_ranking: feature indices ordered by descending importance
    """
    if len(feature_ranking) == 0:
        pred = predict_fn(x).item()
        return 0.0, np.array([0.0]), np.array([pred])

    d = x.shape[1]
    x_del = x.copy()
    scores = [predict_fn(x_del).item()]

    for j in feature_ranking:
        x_del[:, j] = baseline[j]
        scores.append(predict_fn(x_del).item())

    xs = np.linspace(0, len(feature_ranking) / d, len(feature_ranking) + 1)
    auc = np.trapz(scores, xs)
    return auc, xs, np.array(scores)
import numpy as np

def preservation_auc(predict_fn, x, feature_ranking, baseline):
    d = x.shape[1]
    # Convert baseline to a 2D array with the same shape as x
    x_cur = baseline.reshape(1, -1).copy()   # Shape (1, d)
    scores = [predict_fn(x_cur).item()]      # Prediction with no features retained

    for j in feature_ranking:
        x_cur[:, j] = x[0, j]               # Retain the feature by restoring its original value from the baseline
        scores.append(predict_fn(x_cur).item())

    xs = np.linspace(0, len(feature_ranking) / d, len(feature_ranking) + 1)
    auc = np.trapz(scores, xs)
    return auc, xs, np.array(scores)

import numpy as np

def deletion_auc_r(predict_fn, x, feature_ranking, baseline):
    x = np.asarray(x)
    baseline = np.asarray(baseline).reshape(-1)

    assert x.ndim == 2 and x.shape[0] == 1
    assert baseline.shape[0] == x.shape[1]

    d = x.shape[1]
    x_del = x.copy()
    scores = [float(predict_fn(x_del).item())]

    for j in feature_ranking:
        x_del[:, j] = baseline[j]
        scores.append(float(predict_fn(x_del).item()))

    xs = np.linspace(0, len(feature_ranking) / d, len(feature_ranking) + 1)
    auc = np.trapz(scores, xs)
    return auc, xs, np.array(scores)


def preservation_auc_r(predict_fn, x, feature_ranking, baseline):
    x = np.asarray(x)
    baseline = np.asarray(baseline).reshape(-1)
    assert x.ndim == 2 and x.shape[0] == 1
    assert baseline.shape[0] == x.shape[1]

    d = x.shape[1]
    x_cur = baseline.reshape(1, -1).copy()
    scores = [float(predict_fn(x_cur).item())]

    for j in feature_ranking:
        x_cur[:, j] = x[0, j]
        scores.append(float(predict_fn(x_cur).item()))
    xs = np.linspace(0, len(feature_ranking) / d, len(feature_ranking) + 1)
    auc = np.trapz(scores, xs)
    return auc, xs, np.array(scores)


def normalized_preservation_auc(predict_fn, x, feature_ranking, baseline):
    """
    Normalized Preservation AUC.
    Preservation AUC starts from the baseline and restores features one by one;
    predictions should move toward the original prediction.
    - Best case, with a perfect feature ranking: AUC is close to original_area
    - Worst case, with a completely wrong feature ranking: AUC is close to baseline_area
    
    After normalization: 1.0 is best and 0.0 is worst.
    """
    pred_original = float(predict_fn(x).item())
    pred_baseline = float(predict_fn(np.asarray(baseline).reshape(1, -1)).item())

    auc, xs, _ = preservation_auc_r(predict_fn, x, feature_ranking, baseline)
    T = xs[-1]

    # Compute boundary areas
    area_min = min(pred_original, pred_baseline) * T
    area_max = max(pred_original, pred_baseline) * T

    if np.isclose(area_max, area_min):
        return 1.0

    # Normalize: the closer auc is to the area for pred_original, the closer the value is to 1
    normalized = (auc - area_min) / (area_max - area_min)
    
    # If original > baseline, larger auc is better
    # If original < baseline, smaller auc is better, so invert it
    if pred_original < pred_baseline:
        normalized = 1.0 - normalized
    
    # Clamp to the [0, 1] range
    return float(np.clip(normalized, 0.0, 1.0))


def normalized_deletion_auc(predict_fn, x, feature_ranking, baseline):
    """
    Normalized Deletion AUC.
    
    Deletion AUC starts from the original instance and removes features one by one;
    predictions should move toward the baseline prediction.
    - Best case, with a perfect feature ranking: AUC is close to baseline_area
    - Worst case, with a completely wrong feature ranking: AUC is close to original_area
    
    After normalization: 1.0 is best and 0.0 is worst.
    """
    pred_original = float(predict_fn(x).item())
    pred_baseline = float(predict_fn(np.asarray(baseline).reshape(1, -1)).item())

    auc, xs, _ = deletion_auc_r(predict_fn, x, feature_ranking, baseline)
    T = xs[-1]

    # Compute boundary areas
    area_min = min(pred_original, pred_baseline) * T
    area_max = max(pred_original, pred_baseline) * T

    if np.isclose(area_max, area_min):
        return 1.0

    # Normalize: the closer auc is to the area for pred_baseline, the closer the value is to 1
    normalized = (auc - area_min) / (area_max - area_min)
    
    # If original > baseline, smaller auc is better
    # If original < baseline, larger auc is better
    if pred_original > pred_baseline:
        normalized = 1.0 - normalized
    
    # Clamp to the [0, 1] range
    return float(np.clip(normalized, 0.0, 1.0))


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


