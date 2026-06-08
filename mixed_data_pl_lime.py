import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
# === Import external functions ===
from new_select_feature import run, compute_stability
from fit_knots import fit_piecewise_local_model1xx_zb
from evaluate import slope_trend_similarity,multiple_run_consistency,stability_used_feature,deletion_auc_cat1,deletion_auc_r,make_baseline,preservation_auc,normalized_preservation_auc
import pandas as pd
import os
import time
from data_generate import data_inverse_1
from no_onehot_fit_knots import fit_piecewise_local_model_li,find_steps_with_exact_k
from collections import defaultdict
from sklearn.metrics import r2_score

import sklearn

class LocalModelStabilityAnalyzer:
    def __init__(self, predict_fn, X, y, instance,categorical_features,continuous_features,groupk, n0=1, N=50, n_perturbation=6250,scale_factor=1, random_state=42,save_path="results.csv"):
        """
        Initialize the analyzer.

        Parameters:
        ----------
        model : trained model object (must support predict_proba)
        X, y : ndarray, data and labels
        k1 : int, number of selected features
        n0 : int, node-count parameter
        N : int, number of repeated experiments
        scale_factor : float, perturbation standard-deviation scale factor
        """

        self.X = X
        self.y = y

        self.n0 = n0
        self.N = N
        self.n_perturbation = n_perturbation
        self.scale_factor = scale_factor
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = None
        self.instance = instance
        self.save_path = save_path
        self.scale = np.std(X, axis=0)
        self.loc = np.mean(X, axis=0)
        self.categorical_features=categorical_features
        self.continuous_features=continuous_features
        # Initialize storage
        self.results_summary = {}
        self.history = []
        self.predict_fn=predict_fn
        self.groupk=groupk

    # === Step 3: Single experiment ===
    def single_run(self, X_instance):

        r2_group_all = defaultdict(list)
        mse_group_all = defaultdict(list)
        rank_group_all = defaultdict(list)
        use_group_all = defaultdict(list)
        prob_diag_all = []
        kknots, w1, uu_value, kk_set, use_ng,r2_ =  [], [], [], [], [], []
        d_auc_group_all = defaultdict(list)
        for m in range(self.N):
            data1,inverse_data1 = data_inverse_1(self.X,self.instance,self.categorical_features,self.continuous_features,self.scale,num_samples=int(self.n_perturbation/0.8))

            #data1, inverse_data1 = explainer._LimeTabularExplainer__data_inverse(self.instance, num_samples=int(
            #    self.n_perturbation / 0.8))

            #predictions = self.predict_fn(inverse_data1)[:, 0]

            mean = np.zeros(len(self.instance))
            scale = np.ones(len(self.instance))
            mean[self.continuous_features] = self.loc[self.continuous_features]
            scale[self.continuous_features] = self.scale[self.continuous_features]
            data_transformed = (data1) / scale
            distances = pairwise_distances(data_transformed, data_transformed[0].reshape(1,-1)).ravel()

            predictions = self.predict_fn(inverse_data1)[:, 0]
            #predictions = self.predict_fn(inverse_data1)  # Regression problem
            p = predictions

            prob_diag_all.append({
                "mid_ratio": np.mean((p > 0.1) & (p < 0.9)),
                "sat_ratio": np.mean((p < 0.05) | (p > 0.95)),
                "p_std": np.std(p),
                "p_min": np.min(p),
                "p_max": np.max(p),
                "p_mean": np.mean(p)
            })

            kernel_width = np.sqrt(data_transformed.shape[1]) * 0.75
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                data_transformed, predictions, weights, test_size=0.2, random_state=self.random_state
            )

            knots = [np.linspace(np.min(X_train[:, idx]), np.max(X_train[:, idx]), self.n0 + 1)
                     for idx in self.continuous_features]
            n_features = X_train.shape[1]
            n_cont=len(self.continuous_features)
            p_g = np.ones(n_features)
            p_g[:n_cont] = self.n0
            #p_g = np.ones(n_features)
            results = fit_piecewise_local_model_li(p_g,np.ones(n_features),
                X_train, y_train, w_train, self.continuous_features, self.categorical_features,X_test, y_test, w_test,
                knots=knots, n0=self.n0, lambda1=0.00,
            )

            #f_11 = sorted(enumerate(kk), key=lambda x: abs(x[1]), reverse=True)
            origal_order = []

            n_cont=len(self.continuous_features)
            def get_origal_order(id):
                if id <= n_cont-1:
                    return (self.continuous_features[id])
                if id > n_cont-1:
                    return (self.categorical_features[id - n_cont])
            #
            #for i in range(len(self.instance)):
            #     value = get_origal_order(f_11[i][0])
            #     origal_order.append(value)
            #w1.append(origal_order)
            col_stds = results["col_stds"]
            uu_value.append(results["u_value"])
            kknots.append(knots)


            for num in self.groupk:
                steps_k = find_steps_with_exact_k(results["actives"], num, verbose=False)
                # print(steps_k)
                id_k = steps_k[0][0]

                beta_k = results["beta_hat_path"][id_k]
                M_test_k, y_test_k, w_test_k = results["M_xyw"]
                y_pred_k = M_test_k.dot(beta_k)
                r2_test_k = r2_score(y_test_k, y_pred_k, sample_weight=w_test_k)
                mse_test_k = np.average((y_test_k - y_pred_k) ** 2, weights=w_test_k)
                r2_group_all[num].append(r2_test_k)
                mse_group_all[num].append(mse_test_k)

                kk = [beta_k[i] * col_stds[i] for i in range(self.X.shape[1])]
                origal_order = []
                # for i in range(len(self.instance)):
                #     value = get_origal_order(f_11[i][0])
                #     origal_order.append(value)
                # w1.append(origal_order)
                rank_group_all[num].append(kk)  # This is no longer the true ranking; return the true ranking.
                sorted_indices = np.argsort(kk)[::-1]
                sorted_indices_x=[get_origal_order(i) for i in sorted_indices]

                

                used_features = steps_k[0][1]
                use_group_all[num].append(used_features)

        rank_stab = {k: multiple_run_consistency(v) for k, v in rank_group_all.items()}
        use_stab = {k: stability_used_feature(v) for k, v in use_group_all.items()}
        r2_mean = {k: np.mean(v) for k, v in r2_group_all.items()}
        r2_std = {k: np.std(v) for k, v in r2_group_all.items()}
        mse_mean = {k: np.mean(v) for k, v in mse_group_all.items()}
        mse_std = {k: np.std(v) for k, v in mse_group_all.items()}
        
        use_set = defaultdict(list)
        prob_diagnostic = {
            "mid_ratio": np.mean([d["mid_ratio"] for d in prob_diag_all]),
            "sat_ratio": np.mean([d["sat_ratio"] for d in prob_diag_all]),
            "p_std": np.mean([d["p_std"] for d in prob_diag_all]),
            "p_min": np.mean([d["p_min"] for d in prob_diag_all]),
            "p_max": np.mean([d["p_max"] for d in prob_diag_all]),
            "p_mean": np.mean([d["p_mean"] for d in prob_diag_all])
        }
        for k, v in use_group_all.items():
            sets = [set(lst) for lst in v]
            common_elements = set.intersection(*sets)
            continuous_set = set(range(n_cont))
            final_common = common_elements & continuous_set
            use_set[k] = final_common

        return prob_diagnostic,uu_value,kknots,use_set,r2_mean,mse_mean,r2_std,mse_std,rank_stab,use_stab,rank_group_all

    # === Step 4: Main workflow ===
    def run_analysis(self, run_name=None):
        start_time = time.time()
        X_instance = self.instance

        prob_diagnostic,uu_value, kknots, use_set, r2_mean, mse_mean, r2_std, mse_std, rank_stab, use_stab,rank_group_all = self.single_run(X_instance)
        avg_similarity = []
        for i in range(len(self.continuous_features)):  # continuous_features comes first
            yy, xx = [], []
            for j in range(self.N):
                array = uu_value[j][i * self.n0:(i + 1) * self.n0]
                yy.append(np.insert(array, 0, 0))
                xx.append(kknots[j][i])
            avg_similarity.append(slope_trend_similarity(yy, xx))

        u_ = {}
        for k, v in use_set.items():
            if len(v) == 0:
                u_[k] = np.nan
            else:
                u_[k] = np.mean([avg_similarity[i] for i in v])

        end_time = time.time()
        duration = end_time - start_time

        # Record results
        result = {
            "run_name": run_name or f"run_{len(self.history) + 1}",
            "duration_sec": round(duration, 3),
            "used_features_stability": use_stab,
            "ranking_features_stability": rank_stab,
            "shape_slope_stability": u_,
            "r2_mean": r2_mean,
            "mse_mean": mse_mean,
            "r2_std": r2_std,
            "mse_std": mse_std,
            "N": self.N,
            "n_perturbation": self.n_perturbation,
            "scale_factor": self.scale_factor,
            # "self.instance" : self.instance,
            "n0": self.n0,
            "kknots": kknots,
            "uu_value":uu_value,
            "rank_group_all":rank_group_all,  # Transform to obtain the true ranking
            #"w1": w1,
            "prob_diagnostic":prob_diagnostic,
            
        }
        self.history.append(result)
        #print(f"LIME run completed: {result}")
        return result

    def save_results(self):
        df = pd.DataFrame(self.history)
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                df.to_csv(self.save_path, index=False)
            else:
                df.to_csv(self.save_path, mode='a', header=False, index=False)



