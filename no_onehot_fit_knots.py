
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import numpy as np
import cvxpy as cp
from sklearn.metrics import r2_score, mean_squared_error
from new_select_feature import nonnegative_garrotte_path

from new_select_feature import find_steps_with_exact_k,apply_centered_weights_train



def build_mix_matrix(X, continuous_idx, categorical_idx=None, knots=None, n0=None):

    n = X.shape[0]
    A_parts = []
    cat_var_sizes = []
    for i, idx in enumerate(continuous_idx):
        xi = X[:, idx]
        k = knots[i]  # Knot sequence for the i-th continuous feature; continuous features only
        B = np.zeros((n, n0))
        for j in range(n0):
            left, right = k[j], k[j + 1]
            # Linear spline basis function
            B[:, j] = np.clip((xi - left) / (right - left), 0, 1)
        A_parts.append(B)
    if categorical_idx:
        for i, idx in enumerate(categorical_idx):
            X_cat = X[:, idx]
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(-1, 1)
            A_parts.append(X_cat)
            cat_var_sizes.append(X_cat.shape[1])
    A = np.hstack(A_parts)
    return A, cat_var_sizes

def fit_model_full(X_train, y_train, weights_train, continuous_idx,
                                     categorical_idx=None,
                                     X_test=None, y_test=None, weights_test=None,
                                     knots=None, n0=5,
                                     lambda1=0.0, lambda_sparse=0.0, kk=None):
    n, n_feature = X_train.shape
    A,cat_var_sizes = build_mix_matrix(X_train, continuous_idx, categorical_idx=categorical_idx, knots=knots, n0=n0)
    # D matrix and its inverse
    n, _ = X_train.shape
    n_cont = len(continuous_idx) if continuous_idx is not None else 0
    n_cat = len(categorical_idx) if categorical_idx is not None else 0

    D = np.zeros((n_cont * n0, n_cont * n0))
    for i in range(n_cont):
        D[i * n0:(i + 1) * n0, i * n0:(i + 1) * n0] = np.tril(np.ones((n0, n0)))
    Df = np.linalg.inv(D)

    C = np.zeros((n_cont * n0 - 1, n_cont * n0))
    for i in range(n_cont * n0 - 1):
        C[i, i] = -1
        C[i, i + 1] = 1
    rows_to_delete = np.arange(n0 - 1, n_cont * n0 - 1, n0)
    C1 = np.delete(C, rows_to_delete, axis=0)

    A_wmean = np.average(A, axis=0, weights=weights_train)  # sum(w*x)/sum(w)
    y_wmean = np.average(y_train, weights=weights_train)

    A = A - A_wmean
    y_train_c = y_train - y_wmean

    n_basis = A.shape[1]
    w = cp.Variable(n_basis)
    d_u = w[:n_cont * n0]
    u = D @ d_u

    if C1.shape[0] == 0 or C1.shape[1] == 0:
        regularization_term1 = 0
    else:
        regularization_term1 = (lambda1 /  n_cont) * cp.sum_squares(C1 @ u)  # First-order difference

    objective1 = cp.Minimize(
        cp.sum(cp.multiply(weights_train, cp.square(A @ w -y_train_c))) / n+
        regularization_term1
    )
    cp.Problem(objective1).solve(solver=cp.SCS)

    d_u_value = w.value[:n_cont * n0]
    '''
    beta_norm = np.hstack([
        np.linalg.norm(d_u_value[j * n0:(j + 1) * n0])
        for j in range(n_cont)
    ])
    '''
    M_train = np.zeros((n, n_cont + n_cat))
    beta_norm_cont = []
    beta_norm_cat = []
    norm_u_value = []
    beta_max = []
    for j in range(n_cont):
        beta_j = d_u_value[j * n0:(j + 1) * n0]
        #norm_1=np.linalg.norm(d_u_value[j * n0:(j + 1) * n0])
        norm_1 = np.linalg.norm(d_u_value[j * n0:(j + 1) * n0],ord=1)
        M_train[:, j] = (A[:, j * n0:(j + 1) * n0] @ beta_j)
        beta_norm_cont.append(norm_1)
        beta_max.append(np.max(abs(M_train[:, j])))

    cat_start = n_cont * n0
    cur = cat_start
    col_ptr = n_cont

    for group in categorical_idx:
        # If group is a single index, such as 5
        if np.isscalar(group):
            width = 1
            #norm_2 = np.linalg.norm(w.value[cur])
            norm_2 = np.linalg.norm([w.value[cur]],ord=1)
            M_train[:, col_ptr] = w.value[cur]*A[:, cur]
            cur += 1
            col_ptr += 1
            beta_norm_cat.append(norm_2)

        else:
            # group spans multiple columns, such as [5, 6, 7]
            width = len(group)
            M_train[:, col_ptr:col_ptr + width] = w.value[cur]*A[:, cur:cur + width]
            norm_2 = np.linalg.norm(w.value[cur:cur + width],ord=1)
            cur += width
            col_ptr += width
            beta_norm_cat.append(norm_2)

    beta_norm = beta_norm_cont + beta_norm_cat
    r2_train = r2_score(y_train_c, A @ w.value, sample_weight=weights_train)

    col_mean = np.mean(M_train, axis=0)
    col_stds = np.std(M_train, axis=0)
    col_stds[col_stds == 0] = 1e-8

    if X_test is not None and y_test is not None:
        A_test,cat_var_size1 = build_mix_matrix(X_test, continuous_idx, categorical_idx=categorical_idx, knots=knots, n0=n0)
        A_test = A_test - A_wmean
        y_test = y_test - y_wmean
        #print([y_test.shape,(A_test @ w.value).shape])
        r2_test = r2_score(y_test, A_test @ w.value, sample_weight=weights_test)
        mse_test = np.average((y_test - (A_test @ w.value)) ** 2, weights=weights_test)

    results = {
        "u_value": D @ d_u_value,
        "knots": knots,
        "piecewise_parameters": d_u_value,
        "col_stds": col_stds,
        "r2_test": r2_test,
        "mse_test": mse_test,
        "r2_train": r2_train,

    }
    return results

'''
def fit_model_lambda(l_lambda,X_train, y_train, weights_train, continuous_idx,
                                     categorical_idx=None,
                                     X_test=None, y_test=None, weights_test=None,
                                     knots=None, n0=5,
                                      ):
    n, n_feature = X_train.shape
    A,cat_var_sizes = build_mix_matrix(X_train, continuous_idx, categorical_idx=categorical_idx, knots=knots, n0=n0)
    # D matrix and its inverse
    n, _ = X_train.shape
    n_cont = len(continuous_idx) if continuous_idx is not None else 0
    n_cat = len(categorical_idx) if categorical_idx is not None else 0

    A_wmean = np.average(A, axis=0, weights=weights_train)  # sum(w*x)/sum(w)
    y_wmean = np.average(y_train, weights=weights_train)

    A = A - A_wmean
    y_train_c = y_train - y_wmean

    n_basis = A.shape[1]
    w = cp.Variable(n_basis)



    objective1 = cp.Minimize(
        cp.sum(cp.multiply(weights_train, cp.square(A @ w -y_train_c))) / n

    )
    cp.Problem(objective1).solve(solver=cp.SCS)

    d_u_value = w.value[:n_cont * n0]

    M_train = np.zeros((n, n_cont + n_cat))
    beta_norm_cont = []
    beta_norm_cat = []
    norm_u_value = []

    for j in range(n_cont):
        beta_j = d_u_value[j * n0:(j + 1) * n0]
        #norm_1=np.linalg.norm(d_u_value[j * n0:(j + 1) * n0])
        norm_1 = np.linalg.norm(d_u_value[j * n0:(j + 1) * n0],ord=2)
        M_train[:, j] = (A[:, j * n0:(j + 1) * n0] @ beta_j)
        beta_norm_cont.append(norm_1)

    cat_start = n_cont * n0
    cur = cat_start
    col_ptr = n_cont

    for group in categorical_idx:
        # If group is a single index, such as 5
        if np.isscalar(group):
            width = 1
            #norm_2 = np.linalg.norm(w.value[cur])
            norm_2 = np.linalg.norm([w.value[cur]],ord=1)
            M_train[:, col_ptr] = w.value[cur]*A[:, cur]
            cur += 1
            col_ptr += 1
            beta_norm_cat.append(norm_2)

        else:
            # group spans multiple columns, such as [5, 6, 7]
            width = len(group)
            M_train[:, col_ptr:col_ptr + width] = w.value[cur]*A[:, cur:cur + width]
            norm_2 = np.linalg.norm(w.value[cur:cur + width],ord=1)
            cur += width
            col_ptr += width
            beta_norm_cat.append(norm_2)

    beta_norm = beta_norm_cont + beta_norm_cat
    r2_train = r2_score(y_train_c, A @ w.value, sample_weight=weights_train)

    l = sum((np.square(A @ w.value - y_train_c))) / (n - n_basis - 1)
    #print("l=", l)

    b_ng = cp.Variable(n_feature)
    objective2 = cp.Minimize(
        cp.sum(cp.multiply(weights_train, cp.square( M_train @ b_ng - y_train_c))) +
        l_lambda*(cp.sum(n0 * b_ng))
    )

    cp.Problem(objective2).solve(solver=cp.SCS)
    used_feature = np.where(b_ng.value >= 1e-6)[0]

    M_test = np.zeros((n, n_cont + n_cat))
    if X_test is not None and y_test is not None:
        A_test,cat_var_size1 = build_mix_matrix(X_test, continuous_idx, categorical_idx=categorical_idx, knots=knots, n0=n0)
        A_test = A_test - A_wmean
        y_test = y_test - y_wmean
        #print([y_test.shape,(A_test @ w.value).shape])
        M_test = np.zeros((X_test.shape[0], n_cont + n_cat))
        for j in range(n_cont):
            beta_j = d_u_value[j * n0:(j + 1) * n0]
            M_test[:, j] = (A_test[:, j * n0:(j + 1) * n0] @ beta_j)
        r2_test = r2_score(y_test, M_test @ b_ng.value, sample_weight=weights_test)
        mse_test = np.average((y_test - (M_test @ b_ng.value)) ** 2, weights=weights_test)


    results = {

        "knots": knots,
        "piecewise_parameters": d_u_value,
        "r2_test": r2_test,
        "mse_test": mse_test,
        "r2_train": r2_train,
        "used_feature": used_feature,

    }
    return results
'''
import numpy as np
import cvxpy as cp
from sklearn.metrics import r2_score

def compute_lambda_max(M, y, weights, c):
    """
    M: (n, p) design matrix
    y: (n,) target values after centering
    weights: (n,) sample weights
    c: (p,) penalty-weight vector
    """
    # Compute M^T (W y)
    inner = M.T @ (weights * y)   # Shape (p,)
    # Take the positive part
    positive_part = np.maximum(inner, 0.0)
    # Divide by c_i and take the maximum
    lam_max = np.max(2.0 * positive_part / c)
    return lam_max
def make_lambda_grid(M, y, weights, c, n_lambdas=20, ratio=1e-3):
    lam_max = compute_lambda_max(M, y, weights, c)
    lam_min = lam_max * ratio
    lambdas = np.logspace(np.log10(lam_max), np.log10(lam_min), n_lambdas)
    return lambdas
def fit_model_lambda(l_lambda, X_train, y_train, weights_train, continuous_idx,
                     categorical_idx=None,
                     X_test=None, y_test=None, weights_test=None,
                     knots=None, n0=5):

    n, n_feature = X_train.shape
    # Build basis matrix A, including continuous-feature bases and categorical dummy variables
    A, cat_var_sizes = build_mix_matrix(X_train, continuous_idx,
                                        categorical_idx=categorical_idx,
                                        knots=knots, n0=n0)
    n_cont = len(continuous_idx) if continuous_idx is not None else 0
    n_cat = len(categorical_idx) if categorical_idx is not None else 0

    # Weighted centering
    A_wmean = np.average(A, axis=0, weights=weights_train)
    y_wmean = np.average(y_train, weights=weights_train)
    A_centered = A - A_wmean
    y_centered = y_train - y_wmean

    n_basis = A_centered.shape[1]

    # ------------------ First stage: solve w (basis-function coefficients) ------------------
    w = cp.Variable(n_basis)
    objective1 = cp.Minimize(
        cp.sum(cp.multiply(weights_train, cp.square(A_centered @ w - y_centered))) / n
    )
    prob1 = cp.Problem(objective1)
    prob1.solve(solver=cp.SCS)
    if prob1.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: first stage optimization did not converge. Status: {prob1.status}")
    w_opt = w.value
    if w_opt is None:
        raise RuntimeError("First stage optimization failed to produce a solution.")

    # Extract piecewise parameters for continuous features, with n0 parameters per feature
    d_u_value = w_opt[:n_cont * n0]

    # Build M_train as each feature's contribution after the linear combination
    M_train = np.zeros((n, n_cont + n_cat))
    beta_norm_cont = []  # L2 norm for continuous features
    beta_norm_cat = []  # L1 norm for categorical features, absolute value for scalars

    # Continuous features
    for j in range(n_cont):
        beta_j = d_u_value[j * n0:(j + 1) * n0]
        norm_cont = np.linalg.norm(beta_j, ord=2)
        M_train[:, j] = A_centered[:, j * n0:(j + 1) * n0] @ beta_j
        beta_norm_cont.append(norm_cont)

    # Categorical features
    cat_start = n_cont * n0
    cur = cat_start
    col_ptr = n_cont
    for group in categorical_idx:
        if np.isscalar(group):
            # Single categorical variable, such as a one-hot encoded column
            width = 1
            norm_cat = np.linalg.norm([w_opt[cur]], ord=1)
            M_train[:, col_ptr] = w_opt[cur] * A_centered[:, cur]
            cur += 1
            col_ptr += 1
            beta_norm_cat.append(norm_cat)


    beta_norm = beta_norm_cont + beta_norm_cat

    # Training-set R², computed on centered data and equivalent to the original scale
    y_pred_train = A_centered @ w_opt
    r2_train = r2_score(y_centered, y_pred_train, sample_weight=weights_train)

    l = sum((np.square(y_pred_train - y_centered))) / (n - n_basis - 1)
    #l1=l*np.log(n)/2
    #print(l,l1)


    # ------------------ Second stage: solve b_ng (feature-importance weights theta) ------------------
    # Build penalty weights c: continuous-feature weight = n0, categorical-feature weight = 1
    c = np.zeros(n_cont + n_cat)
    c[:n_cont] = n0
    c[n_cont:] = 1.0
    lam_max = compute_lambda_max(M_train, y_centered, weights_train, c)
    #print(f"λ_max = {lam_max}")
    #l_lambda = make_lambda_grid(M_train, y_centered, weights_train, c)#
    l_lambda=np.log10(lam_max)

    b_ng = cp.Variable(n_cont + n_cat, nonneg=True)  # Nonnegative constraint
    objective2 = cp.Minimize(
        cp.sum(cp.multiply(weights_train, cp.square(M_train @ b_ng - y_centered))) +
        l_lambda * cp.sum(cp.multiply(c, b_ng))
    )
    prob2 = cp.Problem(objective2)
    prob2.solve(solver=cp.SCS)

    if prob2.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: second stage optimization did not converge. Status: {prob2.status}")
    b_ng_opt = b_ng.value

    if b_ng_opt is None:
        raise RuntimeError("Second stage optimization failed to produce a solution.")

    col_stds = np.std(M_train, axis=0)


    # ------------------ Test-set evaluation, if provided ------------------
    r2_test = None
    mse_test = None
    if X_test is not None:
        if y_test is None:
            raise ValueError("y_test must be provided when X_test is given.")
        # Build test-set basis matrix A_test
        A_test, _ = build_mix_matrix(X_test, continuous_idx,
                                     categorical_idx=categorical_idx,
                                     knots=knots, n0=n0)
        # Use the training-set centering parameters
        A_test_centered = A_test - A_wmean
        y_test_centered = y_test - y_wmean

        # Build M_test
        M_test = np.zeros((X_test.shape[0], n_cont + n_cat))
        # Continuous-feature part
        for j in range(n_cont):
            beta_j = d_u_value[j * n0:(j + 1) * n0]
            M_test[:, j] = A_test_centered[:, j * n0:(j + 1) * n0] @ beta_j
        # Categorical-feature part
        cur = cat_start
        col_ptr = n_cont
        for group in categorical_idx:
            if np.isscalar(group):
                width = 1
                M_test[:, col_ptr] = w_opt[cur] * A_test_centered[:, cur]
                cur += 1
                col_ptr += 1

        y_pred_test = M_test @ b_ng_opt
        # Compute weighted MSE and R²
        if weights_test is None:
            weights_test = np.ones(X_test.shape[0])
        mse_test = np.average((y_test_centered - y_pred_test) ** 2, weights=weights_test)
        r2_test = r2_score(y_test_centered, y_pred_test, sample_weight=weights_test)

    rank = [b_ng_opt[i] * col_stds[i] for i in range(n_feature)]
    used_feature = np.where(b_ng_opt >= 1e-6)[0]

    results = {
        "knots": knots,
        "piecewise_parameters": d_u_value,
        "w_opt": w_opt,
        "b_ng_opt": b_ng_opt,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mse_test": mse_test,
        "used_feature": used_feature,
        "beta_norm": beta_norm,
        "rank": rank,

    }
    return results



def fit_piecewise_local_model_li(p_g,b_global,X_train, y_train, weights_train,continuous_idx, categorical_idx=None,
                                    X_test=None, y_test=None, weights_test=None,
                                    knots=None, n0=5,
                                    lambda1=0.0, lambda_sparse=0.0, kk=None):
    n, n_feature = X_train.shape

    A,cat_var_sizes = build_mix_matrix(X_train, continuous_idx, categorical_idx=categorical_idx, knots=knots, n0=n0)
    # D matrix and its inverse
    n, _ = X_train.shape
    n_cont = len(continuous_idx) if continuous_idx is not None else 0
    n_cat = len(categorical_idx) if categorical_idx is not None else 0

    D = np.zeros((n_cont * n0, n_cont * n0))
    for i in range(n_cont):
        D[i * n0:(i + 1) * n0, i * n0:(i + 1) * n0] = np.tril(np.ones((n0, n0)))
    Df = np.linalg.inv(D)

    C = np.zeros((n_cont * n0 - 1, n_cont * n0))
    for i in range(n_cont * n0 - 1):
        C[i, i] = -1
        C[i, i + 1] = 1
    rows_to_delete = np.arange(n0 - 1, n_cont * n0 - 1, n0)
    C1 = np.delete(C, rows_to_delete, axis=0)

    A_wmean = np.average(A, axis=0, weights=weights_train)  # sum(w*x)/sum(w)
    y_wmean = np.average(y_train, weights=weights_train)

    A = A - A_wmean
    y_train_c = y_train - y_wmean

    n_basis = A.shape[1]
    w = cp.Variable(n_basis)
    d_u = w[:n_cont * n0]
    u = D @ d_u

    if C1.shape[0] == 0 or C1.shape[1] == 0:
        regularization_term1 = 0
    else:
        regularization_term1 = (lambda1 /  n_cont) * cp.sum_squares(C1 @ u)  # First-order difference



    objective1 = cp.Minimize(
        cp.sum(cp.multiply(weights_train, cp.square(A @ w -y_train_c))) / n+
        regularization_term1
    )
    cp.Problem(objective1).solve(solver=cp.SCS)

    d_u_value = w.value[:n_cont * n0]
    '''
    beta_norm = np.hstack([
        np.linalg.norm(d_u_value[j * n0:(j + 1) * n0])
        for j in range(n_cont)
    ])
    '''
    M_train = np.zeros((n, n_cont + n_cat))
    beta_norm_cont = []
    beta_norm_cat = []
    norm_u_value = []
    beta_max = []
    for j in range(n_cont):
        beta_j = d_u_value[j * n0:(j + 1) * n0]
        #norm_1=np.linalg.norm(d_u_value[j * n0:(j + 1) * n0])
        norm_1 = np.linalg.norm(d_u_value[j * n0:(j + 1) * n0],ord=1)
        M_train[:, j] = (A[:, j * n0:(j + 1) * n0] @ beta_j)
        beta_norm_cont.append(norm_1)
        beta_max.append(np.max(abs(M_train[:, j])))

    cat_start = n_cont * n0
    cur = cat_start
    col_ptr = n_cont

    for group in categorical_idx:
        # If group is a single index, such as 5
        if np.isscalar(group):
            width = 1
            #norm_2 = np.linalg.norm(w.value[cur])
            norm_2 = np.linalg.norm([w.value[cur]],ord=1)
            M_train[:, col_ptr] = w.value[cur]*A[:, cur]
            cur += 1
            col_ptr += 1
            beta_norm_cat.append(norm_2)

        else:
            # group spans multiple columns, such as [5, 6, 7]
            width = len(group)
            M_train[:, col_ptr:col_ptr + width] = w.value[cur]*A[:, cur:cur + width]
            norm_2 = np.linalg.norm(w.value[cur:cur + width],ord=1)
            cur += width
            col_ptr += width
            beta_norm_cat.append(norm_2)

    beta_norm = beta_norm_cont + beta_norm_cat
    r2_train1 = r2_score(y_train_c, A @ w.value, sample_weight=weights_train)

    col_mean = np.mean(M_train, axis=0)
    col_stds = np.std(M_train, axis=0)
    col_stds[col_stds == 0] = 1e-8

    # Second-stage sparse regression
    ll = sum((np.square(A @ w.value - y_train_c))) / (n - n_basis - 1)

    #beta_init = beta_norm * p_g
    beta_init = np.ones(len(beta_norm))
    d_path, r_path, actives, beta_hat_path, beta_init,remove = nonnegative_garrotte_path(
        p_g,M_train, y=y_train_c, beta_init=beta_init, sample_weight=weights_train, max_iter=5000, tol=1e-12, verbose=False,
    )
    results = {
        "u_value": D @ d_u_value,
        "knots": knots,
        "piecewise_parameters": d_u_value,
        "col_stds": col_stds,
        "w":[w.value,A_wmean,y_wmean],
        "beta_norm": beta_norm,
        "actives": actives,
        "beta_hat_path": beta_hat_path,
        "r2_train1":r2_train1
    }
    if X_test is not None and y_test is not None:
        A_test,cat_var_size1 = build_mix_matrix(X_test, continuous_idx, categorical_idx=categorical_idx, knots=knots, n0=n0)
        A_test = A_test - A_wmean
        M_test = np.zeros((X_test.shape[0], n_cont + n_cat))
        for j in range(n_cont):
            beta_j = d_u_value[j * n0:(j + 1) * n0]
            M_test[:, j] = (A_test[:, j * n0:(j + 1) * n0] @ beta_j)#/beta_norm[j]
        cat_start = n_cont * n0
        cur = cat_start
        col_ptr = n_cont
        for group in categorical_idx:
            if np.isscalar(group):
                width = 1
                M_test[:, col_ptr] = (w.value[cur] * A_test[:, cur])#/beta_norm[col_ptr]
                cur += 1
                col_ptr += 1
            else:
                # group spans multiple columns, such as [5, 6, 7]
                width = len(group)
                M_test[:, col_ptr:col_ptr + width] = (w.value[cur] * A_test[:, cur:cur + width])#/beta_norm[col_ptr]
                cur += width
                col_ptr += width

        results.update({
            "M_xyw":[M_test,y_test - y_wmean,weights_test]
        })
        def predict_fn(X, y=None, weights=None,n_save=2):

            # ===== Build A =====
            A, _ = build_mix_matrix(
                X,
                continuous_idx,
                categorical_idx=categorical_idx,
                knots=knots,
                n0=n0,
            )
            A = A - A_wmean

            # ===== Build M =====
            M = np.zeros((X.shape[0], n_cont + n_cat))

            # Continuous variables
            for j in range(n_cont):
                beta_j = d_u_value[j * n0:(j + 1) * n0]
                M[:, j] = A[:, j * n0:(j + 1) * n0] @ beta_j

            # Categorical variables
            cur = n_cont * n0
            col_ptr = n_cont
            for group in categorical_idx:
                if np.isscalar(group):
                    M[:, col_ptr] = w.value[cur] * A[:, cur]
                    cur += 1
                    col_ptr += 1
                else:
                    width = len(group)
                    M[:, col_ptr:col_ptr + width] = (
                            w.value[cur] * A[:, cur:cur + width]
                    )
                    cur += width
                    col_ptr += width

            # ===== Final prediction (second stage) =====
            y_pred = M.dot(beta_hat_path[n_save])


            out = {
                "y_pred": y_pred+y_wmean,
                "M": M,
                "beta_hat_path":beta_hat_path
            }

            # ===== Optional evaluation =====
            if y is not None:
                y_c = y - y_wmean
                out["r2_1"] = r2_score(y_c, A@ w.value, sample_weight=weights)
                out["r2"] = r2_score(y_c, y_pred, sample_weight=weights)
                out["mse"] = np.average(
                    (y_c - y_pred) ** 2,
                    weights=weights
                )

            return out
        results.update({
            "predict": predict_fn,
        })

    return results

