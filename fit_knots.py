
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

from new_select_feature import select_topk_features_by_path,select_features_with_lasso_path


def build_A(X, knots, n0):
    n, d = X.shape
    A = np.zeros((n, d * n0))
    for i in range(d):
        xi = X[:, i]
        for j in range(n0):
            left, right = knots[i][j], knots[i][j + 1]
            A[:, i * n0 + j] = np.clip((xi - left) / (right - left), 0, 1)
    return A


def fit_piecewise_local_model1xx_zb(X_train, y_train, weights_train=None,
                                    X_test=None, y_test=None, weights_test=None,
                                    knots=None, n0=5,
                                    lambda1=0.0, lambda2=0.0, lambda_sparse=0.0, kk=None):
    n, n_feature = X_train.shape
    A = build_A(X_train, knots, n0)
    if weights_train is None:
        weights_train = np.ones(n)

    # D matrix and its inverse
    D = np.zeros((n_feature * n0, n_feature * n0))
    for i in range(n_feature):
        D[i * n0:(i + 1) * n0, i * n0:(i + 1) * n0] = np.tril(np.ones((n0, n0)))
    Df = np.linalg.inv(D)

    # C1 difference penalty
    C = np.zeros((n_feature * n0 - 1, n_feature * n0))
    for i in range(n_feature * n0 - 1):
        C[i, i] = -1
        C[i, i + 1] = 1
    rows_to_delete = np.arange(n0 - 1, n_feature * n0 - 1, n0)

    C1 = np.delete(C, rows_to_delete, axis=0)

    A_means = np.mean(A, axis=0)
    A = A - A_means
    y_mean = np.mean(y_train)
    y_train_c = y_train - y_mean

    u = cp.Variable(n_feature * n0)
    u1 = Df @ u
    if C1.shape[0] == 0 or C1.shape[1] == 0:
        regularization_term1 = 0
    else:
        regularization_term1 = (lambda1 / n_feature) * cp.sum_squares(C1 @ u)

    objective1 = cp.Minimize(
        #cp.sum(cp.square(A @ u1 - y_train_c)) / n +
        cp.sum(cp.multiply(weights_train, cp.square(A @ u1 - y_train_c))) / n+
        regularization_term1 +
        (lambda2 / n_feature) * cp.sum([
            cp.norm(u1[i * n0:(i + 1) * n0], 2) * n0 for i in range(n_feature)
        ])
    )
    cp.Problem(objective1).solve(solver=cp.SCS)

    beta_initial = u.value
    u1_val = Df @ beta_initial  # shape: (n_feature * n0,); contains difference values
    beta_norm = np.hstack([
        np.linalg.norm(u1_val[j * n0:(j + 1) * n0])
        for j in range(n_feature)
    ])

    M_train = np.zeros((n, n_feature))
    for j in range(n_feature):
        beta_j = u1_val[j * n0:(j + 1) * n0]
        M_train[:, j] = A[:, j * n0:(j + 1) * n0] @ beta_j

    r2_train1 = r2_score(y_train_c, A @ u1_val, sample_weight=weights_train)

    col_mean = np.mean(M_train, axis=0)
    col_stds = np.std(M_train, axis=0)
    col_stds[col_stds == 0] = 1e-8

    # Second-stage sparse regression
    ll = sum((np.square(A @ Df @ u.value - y_train_c))) / (n - n_feature - 1)

    # beta_init=beta_norm
    beta_init = np.ones(n_feature)
    d_path, r_path, actives, beta_hat_path, d_final, beta_init, beta_final ,r2= nonnegative_garrotte_path(
        M_train, y=y_train_c, beta_init=beta_init, sample_weight=weights_train, max_iter=5000, tol=1e-12, verbose=False,
        num_save=kk
    )
    used_features = select_topk_features_by_path(actives, k=kk)

    # X_weighted, y_weighted = apply_sample_weights(M_train, y1,sample_weight=weights_train)

    # X_weighted, y_weighted = M_train, y1
    y_pred = M_train.dot(beta_final)
    r2_train = r2_score(y_train_c, y_pred, sample_weight=weights_train)
    y_pred_orig = M_train.dot(beta_final)


    # If a test set is provided, process it
    results = {
        "sparse_weights": d_final,
        "u_value": u.value,
        "knots": knots,
        "piecewise_parameters": u1_val,
        "col_stds": col_stds,
        "r2_train": r2_train,
        "r2_train1": r2_train1,
        "used_features": used_features,

    }

    if X_test is not None and y_test is not None:
        A_test = build_A(X_test, knots, n0)
        A_test = A_test - A_means
        M_test = np.zeros((X_test.shape[0], n_feature))
        for j in range(n_feature):
            beta_j = u1_val[j * n0:(j + 1) * n0]
            M_test[:, j] = A_test[:, j * n0:(j + 1) * n0] @ beta_j  #
        y_pred = M_test.dot(beta_final)
        r2_test = r2_score(y_test - y_mean, y_pred,sample_weight=weights_test)
        # re=corrected_evaluation(M_test, y_test, weights_test, beta_final,b.value)
        # r2_test = re['weighted_r2']

        results.update({
            "r2_test": r2_test,
            "beta_norm": beta_norm,
            "id": id

        })
    y_train_orig = y_train
    y_train_pred_orig = y_pred_orig + y_mean

    mse_train_orig = np.average((y_train_orig - y_train_pred_orig)**2, weights=weights_train)
    r2_train_orig =  r2_score(y_train, y_train_pred_orig, sample_weight=weights_train)
    results.update({
        "mse_train_orig": mse_train_orig,
        "r2_train_orig": r2_train_orig,
    })

    if X_test is not None and y_test is not None:
        y_test_pred_orig = y_pred + y_mean
        mse_test_orig = np.average((y_test - y_test_pred_orig)**2, weights=weights_test)
        r2_test_orig =  r2_score(y_test, y_test_pred_orig ,sample_weight=weights_test)
        results.update({
            "mse_test_orig": mse_test_orig,
            "r2_test_orig": r2_test_orig,
        })
    return results

