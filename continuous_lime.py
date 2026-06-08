import numpy as np
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.linear_model import Ridge
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import truncnorm
from sklearn.preprocessing import StandardScaler
from new_select_feature import select_features_with_lasso_path
from evaluate import stability_used_feature, multiple_run_consistency,slope_trend_similarity,deletion_auc1,preservation_auc,normalized_preservation_auc,normalized_deletion_auc
from sklearn.metrics import pairwise_distances
import pandas as pd
import time
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import defaultdict


class LimeStabilityAnalyzer:
    def __init__(self, kw,model, X, y,instance, feature_names, class_names,isscaler=False,
                 groupk=[10], N=50, n_perturbation=5000,
                 save_path="lime_results.csv", random_state=42):

        self.model = model
        self.X = X
        self.y = y
        self.isscaler=isscaler
        self.feature_names = feature_names
        self.class_names = class_names
        self.groupk = groupk
        self.N = N
        self.n_perturbation = n_perturbation

        self.save_path = save_path
        self.random_state = random_state
        self.history = []

        # Data standardization
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.scale = np.std(X, axis=0)
        self.loc = np.mean(X, axis=0)
        self.instance=instance
        self.kw=kw

    # === Generate perturbation samples ===

    def generate_continuous_perturbations(self, loc, scale, num_samples=2000):
        noise = np.random.normal(0, scale=scale, size=(num_samples, self.X.shape[1]))
        perturbations = loc + noise
        return perturbations
    def generate_perturbations_with_truncnorm(self, loc, num_samples=6000, trunc_a=-2, trunc_b=2):
        d = self.X.shape[1]
        perturbations = np.zeros((num_samples, d))
        for i in range(d):
            mean_i = loc[i]
            std_i = self.scale[i]
            if std_i == 0:
                std_i = 1e-6
            samples = truncnorm.rvs(trunc_a, trunc_b, loc=mean_i, scale=std_i, size=num_samples)
            perturbations[:, i] = samples


        return perturbations

    def generate_uni_perturbations(self, loc, scale, num_samples=2000):
        # Uniform distribution with the same variance as the original normal distribution
        half_range = np.sqrt(3) * scale
        perturbations = np.random.uniform(loc - half_range, loc + half_range,
                                          size=(num_samples, self.X.shape[1]))
        return perturbations

    def generate_continuous(self, loc, x_train, num_samples=2000):
        # Compute the covariance matrix of x_train
        cov_matrix = np.cov(x_train, rowvar=False)

        # Generate perturbation samples from the covariance matrix
        noise = np.random.multivariate_normal(np.zeros(x_train.shape[1]), cov_matrix, size=num_samples)

        # Add the location offset loc
        perturbations = loc + noise
        return perturbations

    # === Single LIME run ===
    def run_single(self, X_instance):


        r2_group_all = defaultdict(list)
        mse_group_all = defaultdict(list)
        rank_group_all = defaultdict(list)
        coef_all = defaultdict(list)
        use_group_all = defaultdict(list)
        hit_group_all = defaultdict(list)
        d_auc_group_all_p=defaultdict(list)
        d_auc_group_all_d = defaultdict(list)

        for i in range(self.N):
            inverse_data = self.generate_perturbations_with_truncnorm(loc=X_instance,
                                                                      num_samples=int(self.n_perturbation / 0.8))
            #inverse_data = self.generate_uni_perturbations(loc=X_instance, scale=self.scale,
                                                                  #num_samples=int(self.n_perturbation / 0.8))
            #inverse_data = self.generate_continuous(loc=X_instance, x_train=self.X,
                                                                #num_samples=int(self.n_perturbation / 0.8))

          
            denominator = self.scale
            denominator = np.where(denominator == 0, 1, denominator) 

            scaled_data = (inverse_data-self.loc) / denominator

            instance0 = (X_instance.reshape(1, -1)-self.loc) / denominator
            distances = pairwise_distances(scaled_data, instance0).ravel()
            if self.isscaler:
                labels = self.model.predict_proba(scaled_data)[:, 0]
            else:
                labels = self.model.predict_proba(inverse_data)[:, 0]

            kernel_width = np.sqrt(inverse_data.shape[1]) * self.kw
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

            X_train_p, X_test_p, y_train_p, y_test_p, w_train, w_test = train_test_split(
                scaled_data, labels, weights, test_size=0.2, random_state=self.random_state
            )
            col_std=np.std(X_train_p,axis=0)

            for num in self.groupk:
                used_features = select_features_with_lasso_path(X_train_p, y_train_p, w_train, num)
                model = Ridge(alpha=1, fit_intercept=True)
                model.fit(X_train_p[:, used_features], y_train_p, sample_weight=w_train)
                preds = model.predict(X_test_p[:, used_features])
                r2 = r2_score(y_test_p, preds, sample_weight=w_test)
                mse = np.average((y_test_p - preds) ** 2, weights=w_test)
                use_group_all[num].append(used_features)
                r2_group_all[num].append(r2)
                mse_group_all[num].append(mse)
                coef_abs_full = np.zeros(len(self.feature_names))
                coef_full= np.zeros(len(self.feature_names))


                coef_abs_full[used_features] = np.abs(model.coef_)
                coef_full[used_features] = model.coef_

                sorted_indices = np.argsort(coef_abs_full)[::-1]
                predict_fn = lambda x: self.model.predict_proba(x)[0, 0]

                
                coef_all[num].append(coef_full)
                rank_group_all[num].append(coef_abs_full*col_std)  
                hit_group_all[num].append(len(set(used_features) & set(range(4))))

        rank_stab = {k: multiple_run_consistency(v) for k, v in rank_group_all.items()}
        use_stab = {k: stability_used_feature(v) for k, v in use_group_all.items()}
        r2_mean = {k: np.mean(v) for k, v in r2_group_all.items()}
        r2_std = {k: np.std(v) for k, v in r2_group_all.items()}
        mse_mean = {k: np.mean(v) for k, v in mse_group_all.items()}
        mse_std = {k: np.std(v) for k, v in mse_group_all.items()}
        hit_mean = {k: np.mean(v) for k, v in hit_group_all.items()}
        coef_mean = {k: np.mean(v,axis=0) for k, v in coef_all.items()}
        
        rank_mean = {k: np.mean(v, axis=0) for k, v in rank_group_all.items()}
        rank_std = {k: np.std(v, axis=0) for k, v in rank_group_all.items()}

        return rank_std,rank_mean ,col_std,coef_mean,rank_group_all,hit_mean,r2_mean,r2_std ,mse_mean,mse_std,rank_stab,use_stab,

    # === Main analysis entry point ===
    def run_analysis(self, run_name=None):
        start_time =  time.time()
        X_instance = self.instance
        rank_std,rank_mean ,col_std,coef_mean,rank_group_all,hit_mean,r2_mean, r2_std, mse_mean, mse_std, rank_stab, use_stab = self.run_single(X_instance)

        end_time = time.time()  #
        duration = end_time - start_time

        result = {
            "run_name": run_name or f"lime_run_{len(self.history)+1}",
            "duration_sec": round(duration, 3),
            "used_features_stability": use_stab,
            "ranking_features_stability": rank_stab,
            "avg_mse_test": mse_mean,
            "avg_r2_test": r2_mean,
            "avg_mse_std": mse_std,
            "avg_r2_std": r2_std,
            "N": self.N,
            "hit_mean":hit_mean,
            "rank_group_all":rank_group_all,
            "coef_mean":coef_mean,
            #"rank_mean":rank_mean,
            #"rank_std":rank_std,
            #"col_std":col_std
            #"n_perturbation" : self.n_perturbation,

        }

        self.history.append(result)
        #print(f"LIME run completed: {result}")
        return result

    # === Save all run results ===
    def save_results(self):
        df = pd.DataFrame(self.history)
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                df.to_csv(self.save_path, index=False)
            else:
                df.to_csv(self.save_path, mode='a', header=False, index=False)

