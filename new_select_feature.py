# Existing NG solution path and feature selection
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import lars_path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge


def apply_centered_weights_train(X, y, sample_weight=None, eps_std=1e-12):
    """
    Apply sample weights to X and y and standardize features.
    Rules:
    - If sample_weight is None: use ordinary centering + standardization (ddof=1).
    - If sample_weight is provided: first compute the weighted mean and variance
      using w without double weighting, scale features by the weighted standard deviation,
      then multiply X and y by sqrt(w) so ordinary least squares remains equivalent.
    Returns:
      X_processed, y_processed
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    if sample_weight is None:
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        # Sample standard deviation (ddof=1)
        X_std = np.std(X_centered, axis=0, ddof=1)
        X_std[X_std < eps_std] = 1.0
        X_scaled = X_centered

        y_centered = y - np.mean(y)
        return X_scaled, y_centered, X_mean, np.mean(y)

    # Weighted case
    w = np.asarray(sample_weight, dtype=float)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError("sample_weight 长度必须等于样本数")
    if np.any(w < 0):
        raise ValueError("样本权重不能为负")

    sum_w = np.sum(w)
    if sum_w <= 0:
        raise ValueError("样本权重总和必须为正")

    # Weighted mean
    X_wmean = np.average(X, axis=0, weights=w)  # sum(w*x)/sum(w)
    y_wmean = np.average(y, weights=w)

    X_centered = X - X_wmean
    y_centered = y - y_wmean

    X_scaled = (X_centered) * np.sqrt(w)[:, None]
    y_scaled = y_centered * np.sqrt(w)

    return X_scaled, y_scaled, X_wmean, y_wmean


def apply_centered_weights_test(X_test, y_test, X_mean, y_mean=None, sample_weight=None):
    X_test = np.asarray(X_test, dtype=float)
    n_test, p = X_test.shape

    X_scaled = (X_test - X_mean)
    y_scaled = y_test - y_mean

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        if w.ndim != 1 or w.shape[0] != n_test:
            raise ValueError("sample_weight 长度必须等于测试样本数")
        X_scaled = X_scaled * np.sqrt(w)[:, None]
        y_scaled = y_scaled * np.sqrt(w)
    return X_scaled, y_scaled


def compute_enter_alphas(p_g,Z, r, gamma, active_set, tol=1e-12):
    """
    Compute alpha_enter for inactive features (Section 3.1, Step 4).
    Logic: a new feature enters when its residual correlation matches that of the active features;
    the formula strictly matches the document.
    """
    p = Z.shape[1]
    enter_alphas = {}
    if not active_set:
        return enter_alphas

    j_ref = active_set[0]  # Reference feature in the active set (j* in the document); may not satisfy arbitrary choice
    zr_ref = Z[:, j_ref].dot(r)/p_g[j_ref]
    zzg_ref = Z[:, j_ref].dot(Z.dot(gamma))/p_g[j_ref]

    for j in range(p):
        # if j in active_set or np.abs(gamma[j]) < tol:
        if j in active_set:
            continue

        # Document formula: alpha_j = (Z_j^T r[k-1] - Z_j'^T r[k-1]) / (Z_j^T (Z gamma) - Z_j'^T (Z gamma))
        zr_j = Z[:, j].dot(r)/p_g[j]
        zzg_j = Z[:, j].dot(Z.dot(gamma))/p_g[j]
        numerator = zr_j - zr_ref
        denominator = zzg_j - zzg_ref

        # Fix: numerator and denominator must have the same sign and exceed tol in absolute value, ensuring alpha_j > 0
        if (abs(denominator) > tol) and (abs(numerator) > tol):
            if (numerator > 0 and denominator > 0) or (numerator < 0 and denominator < 0):
                alpha_j = (numerator) / (denominator)  # Avoid sign ambiguity; divide directly
                enter_alphas[j] = alpha_j
    return enter_alphas


def compute_zero_alphas(d, gamma, active_set, tol=1e-12):
    """
    Correct exit step-size calculation.
    """
    zero_alphas = {}
    for j in active_set:
        if gamma[j] < -tol:
            alpha_j = -d[j] / gamma[j]
            if alpha_j > tol and alpha_j <= 1.0 + tol:
                zero_alphas[j] = alpha_j
    return zero_alphas


def nonnegative_garrotte_path(p_g,
        X, y, beta_init=None, sample_weight=None,
        max_iter=1000, tol=1e-12, verbose=True):
    """
    Fixed version: ensures the step size is computed correctly and avoids early termination.
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Input validation
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    if len(y) != n:
        raise ValueError(f"X样本数({n})与y样本数({len(y)})不匹配")

    # Initial estimate
    if beta_init is None:
        X_w_init, y_w_init, _, _ = apply_centered_weights_train(X, y, sample_weight)
        lr_init = LinearRegression(fit_intercept=False)
        #lr_init = Ridge(alpha=0.1, fit_intercept=False)
        lr_init.fit(X_w_init, y_w_init)
        beta_init = lr_init.coef_
    else:
        beta_init = np.asarray(beta_init, dtype=float)
        if len(beta_init) != p:
            raise ValueError(f"beta_init长度({len(beta_init)})与特征数({p})不匹配")

    # Build the Z matrix
    X_weighted, y_weighted, _, _ = apply_centered_weights_train(X, y, sample_weight)
    # X_weighted, y_weighted = X, y

    Z = X_weighted * beta_init[np.newaxis, :]

    # Initialize
    d = np.zeros(p)
    r = y_weighted.copy()
    k = 0  # Factor is k=1

    d_path = [d.copy()]
    r_path = [r.copy()]
    actives = [[]]
    beta_hat_path = [(beta_init * d).copy()]
    cov_zr = Z.T.dot(r)/p_g
    j_star = int(np.argmax(cov_zr))

    C_new = sorted(list(set([j_star])))
    actives.append(C_new)
    removed=[]

    while k < max_iter:
        C = C_new
        # Step 3: compute direction
        gamma = np.zeros(p)
        if len(C) > 0:
            ZC = Z[:, C]
            G = ZC.T.dot(ZC)
            ztr = ZC.T.dot(r)

            try:
                gamma_C = np.linalg.solve(G, ztr)
            except np.linalg.LinAlgError:
                gamma_C = np.linalg.pinv(G).dot(ztr)

            for idx, val in zip(C, gamma_C):
                gamma[idx] = val

        if np.linalg.norm(gamma) < tol:
            if verbose:
                print(f"Iter {k + 1}: 方向向量模长({np.linalg.norm(gamma):.6f})<tol，终止")
            break

        # Step 4-5: compute step sizes using the fixed functions
        enter_alphas = compute_enter_alphas(p_g,Z, r, gamma, C, tol)
        zero_alphas = compute_zero_alphas(d, gamma, C, tol)

        # Step 6: choose the smallest step size
        candidates = [('enter', a, j) for j, a in enter_alphas.items()] + \
                     [('zero', a, j) for j, a in zero_alphas.items()] + \
                     [('full', 1.0, None)]
        candidates = [c for c in candidates if c[1] > tol]

        if not candidates:
            alpha = 1.0
            kind = 'full'
            idx = None
        else:
            kind, alpha, idx = min(candidates, key=lambda x: x[1])
            alpha = min(alpha, 1.0)

        # Update parameters
        d_new = d + alpha * gamma
        d_new[d_new < 0] = 0.0

        # Update the active set
        if kind == 'enter':
            C_new = sorted(list(set(C + [idx])))
        elif kind == 'zero':
            C_new = sorted([j for j in C if j != idx])
            removed.append(idx)
        else:
            C_new = sorted(list(np.where(d_new > tol)[0]))

        # Step 7: update residuals
        r_new = y_weighted - Z.dot(d_new)
        d = d_new
        r = r_new
        d_path.append(d.copy())
        r_path.append(r.copy())
        actives.append(C_new)
        beta_hat_path.append((beta_init * d).copy())
        if abs(alpha - 1.0) < tol:
            if verbose:
                print("α=1，路径收缩完成，算法终止。")
            break

        if verbose:
            print(f"Iter {k + 1}: 类型={kind:6s}, 步长={alpha:.4g}, 活跃集大小={len(C_new)}")

        k += 1


    if k >= max_iter and verbose:
        print(f"达到最大迭代次数({max_iter})")


    return d_path, r_path, actives, beta_hat_path, beta_init, removed

'''
def select_topk_features_by_path(actives, k):
    """Select the top k features by Garrotte solution-path entry order."""
    selected = []
    for t in range(1, len(actives)):
        new_feats = set(actives[t]) - set(actives[t - 1])
        for j in new_feats:
            if j not in selected:
                selected.append(j)
            if len(selected) >= k:
                return selected
    return selected
'''

'''
def select_topk_features_by_path(actives, k):
    """Select the top k features by Garrotte solution-path entry order."""
    selected = []
    seen = set()

    # Iterate over all active sets and record first appearances in chronological order
    for active_set in actives:
        for feature in active_set:
            if feature not in seen:
                seen.add(feature)
                selected.append(feature)
                if len(selected) >= k:
                    return selected
    return selected
'''
def find_steps_with_exact_k(actives, k, verbose=False):
    """
    Find all steps where the active-set size is exactly k,
    and determine whether that occurrence is unique.

    Returns
    ----
    steps_k : list of (step_idx, active_set)
    is_unique : bool
        Whether it appears exactly once.
    """
    steps_k = [
        (t, active_set)
        for t, active_set in enumerate(actives)
        if len(active_set) == k
    ]

    is_unique = (len(steps_k) == 1)

    if verbose:
        if len(steps_k) == 0:
            print(f"[Info] 路径中未出现 |C| = {k}")
        elif is_unique:
            print(f"[Info] |C| = {k} 只出现 1 次（step {steps_k[0][0]}）")
        else:
            print(f"[Warning] |C| = {k} 出现 {len(steps_k)} 次")

    return steps_k

def select_topk_features_by_path(actives, k,removed):
    """Select the top k features by Garrotte solution-path entry order."""
    selected = []
    seen = set()

    # Iterate over all active sets and record first appearances in chronological order
    for active_set in actives:
        for feature in active_set:
            if feature not in seen:
                seen.add(feature)
                selected.append(feature)
                if len(selected) >= k:
                    return selected
    if set(selected) & set(removed) != set():
        print("used feature removed")
    return selected
# Existing LASSO solution path and feature selection
import numpy as np
from sklearn.linear_model import lars_path

def generate_lars_path(weighted_data, weighted_labels):
    """Generates the lars path for weighted data.

    Args:
        weighted_data: data weighted by kernel
        weighted_labels: labels weighted by kernel

    Returns:
        (alphas, coefs): arrays for regularization parameter and coefficients
    """
    alphas, _, coefs = lars_path(weighted_data,
                                 weighted_labels,
                                 method='lasso',
                                 verbose=False)
    return alphas, coefs


def select_features_with_lasso_path(data, labels, weights, num_features):
    """Selects features using weighted LASSO path.

    Args:
        data: (n_samples, n_features) input matrix
        labels: (n_samples,) target values
        weights: (n_samples,) kernel weights
        num_features: number of features to keep

    Returns:
        used_features: indices of selected features
    """
    # 1. Weighted data and labels

    weighted_data = ((data - np.average(data, axis=0, weights=weights))
                     * np.sqrt(weights[:, np.newaxis]))
    weighted_labels = ((labels - np.average(labels, weights=weights))
                       * np.sqrt(weights))

    # 2. Call LARS path
    _, coefs = generate_lars_path(weighted_data, weighted_labels)

    # 3. Find the last coefficient solution in the LARS path that satisfies num_features
    nonzero = range(weighted_data.shape[1])
    for i in range(len(coefs.T) - 1, 0, -1):
        nonzero = coefs.T[i].nonzero()[0]
        if len(nonzero) <= num_features:
            break

    used_features = nonzero

    return used_features

def select_features_with_lasso_path1(data, labels, weights, num_features):
    """Select features using weighted LASSO path while preserving entry order."""

    # 1. Weighted centering
    weighted_data = ((data - np.average(data, axis=0, weights=weights))
                     * np.sqrt(weights[:, np.newaxis]))
    weighted_labels = ((labels - np.average(labels, weights=weights))
                       * np.sqrt(weights))

    # 2. LARS path
    _, coefs = generate_lars_path(weighted_data, weighted_labels)
    # coefs shape: (n_features, n_steps)

    n_features, n_steps = coefs.shape

    # 3. Record the first path step where each feature enters
    entry_step = {}
    for j in range(n_features):
        nonzero_steps = np.where(coefs[j, :] != 0)[0]
        if len(nonzero_steps) > 0:
            entry_step[j] = nonzero_steps[0]

    # 4. Sort by entry order
    ordered_features = sorted(entry_step.keys(),
                              key=lambda j: entry_step[j])

    # 5. Truncate to the requested number
    used_features = ordered_features[:num_features]

    return np.array(used_features)

import numpy as np
from sklearn.linear_model import lars_path  # Standard sklearn interface for generating the LARS path


def lasso_path_coef(data, labels, weights=None, num_features=30):
    if weights ==None:
        weights = np.ones(len(labels))

    _, coefs = generate_lars_path(data,labels)

    nonzero = range(data.shape[1])
    for i in range(len(coefs.T) - 1, 0, -1):
        nonzero = coefs.T[i].nonzero()[0]
        if len(nonzero) <= num_features:

            break
    selected_coef = coefs.T[i]
    return selected_coef

def jaccard_similarity(set1, set2):
    """Compute the Jaccard similarity of two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def get_top_features(exp_map, top_n=5):
    """Get the top_n feature indices from a LIME / SHAP / LASSO explanation."""

    # Ensure exp_map has the [(index, importance)] format
    if isinstance(exp_map, dict):
        exp_map = list(exp_map.items())  # Convert to a list

    sorted_explanation = sorted(
        exp_map,
        key=lambda x: abs(x[1]),  # Sort by absolute value
        reverse=True
    )

    # Extract the top_n feature indices
    top_features = {x[0] for x in sorted_explanation[:top_n]}

    return top_features


def run(fim, top_n=5, num=10):
    top_features_list = []

    for i in range(num):
        top_features = get_top_features(fim[i], top_n)
        top_features_list.append(top_features)

    return top_features_list


# Compute the stability metric
def compute_stability(top_features_list):
    n = len(top_features_list)
    jaccard_scores = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_similarity(top_features_list[i], top_features_list[j])
            jaccard_scores.append(sim)

    avg_jaccard = np.mean(jaccard_scores)
    return avg_jaccard, jaccard_scores

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import pairwise_distances

def get_sample_ae(DenoisingAutoencoder,x, n_samples,random_state=None,kernel_width=1) :
    n_features=len(x)
    rng = np.random.default_rng(random_state)

    x_embedding = DenoisingAutoencoder.transform(x.reshape(1, -1))
    sample_pool = rng.normal(0.0, 1.0, size=(20000, n_features))
    # Is this being standardized?
    pool_embedding = DenoisingAutoencoder.transform(sample_pool)

    distances = pairwise_distances(pool_embedding, x_embedding).ravel()
    nearest = np.argpartition(distances, kth=n_samples - 1)[:n_samples]
    local_x = sample_pool[nearest]
    local_distances = distances[nearest]
    weights = np.exp(-local_distances / kernel_width)
    return local_x,weights
