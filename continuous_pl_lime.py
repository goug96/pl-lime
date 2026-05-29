import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from no_onehot_fit_knots import fit_piecewise_local_model_li
from evaluate import slope_trend_similarity,multiple_run_consistency,stability_used_feature,deletion_auc1,preservation_auc,normalized_preservation_auc,normalized_deletion_auc
import pandas as pd
import os
import time
from collections import defaultdict
from sklearn.metrics import r2_score
from new_select_feature import find_steps_with_exact_k



class LocalModelStabilityAnalyzer:
    def __init__(self, kw,b_global,model, X, y, instance, isscaler=False,groupk=[10,20,30],n0=1, N=50, n_perturbation=6250,scale_factor=1, random_state=42,save_path="results.csv"):

        self.model = model
        self.X = X
        self.y = y

        self.n0 = n0
        self.N = N
        self.n_perturbation = n_perturbation
        self.scale_factor = scale_factor
        self.random_state = random_state
        self.isscaler = isscaler
        self.kw = kw
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = None
        self.instance = instance
        self.save_path = save_path
        self.scale = np.std(X, axis=0)
        self.loc = np.mean(X, axis=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        sigma = np.cov(X_scaled, rowvar=False)
        self.sigma = sigma
        self.b_global = b_global
        self.groupk = groupk

        # Initialize storage
        self.results_summary = {}
        self.history = []


    def generate_continuous_perturbations(self, loc, scale, num_samples=2000):
        noise = np.random.normal(0, scale=scale, size=(num_samples, self.X.shape[1]))
        perturbations = loc + noise
        return perturbations
    def generate_perturbations_with_truncnorm(self, loc, num_samples=6000, trunc_a=-2, trunc_b=2):
        d = self.X.shape[1]
        perturbations = np.zeros((num_samples, d))
        for i in range(d):
            mean_i = loc[i]
            std_i = self.scale[i] * self.scale_factor
            if std_i == 0:
                std_i = 1e-6
            samples = truncnorm.rvs(trunc_a, trunc_b, loc=mean_i, scale=std_i, size=num_samples)
            perturbations[:, i] = samples
        return perturbations

    import numpy as np

    def generate_continuous(self, loc, x_train, num_samples=2000):
        # Compute the covariance matrix of x_train
        cov_matrix = np.cov(x_train, rowvar=False)

        # Generate perturbation samples from the covariance matrix
        noise = np.random.multivariate_normal(np.zeros(x_train.shape[1]), cov_matrix, size=num_samples)

        # Add the location offset loc
        perturbations = loc + noise
        return perturbations
    def generate_uni_perturbations(self, loc, scale, num_samples=2000):
        # Uniform distribution with the same variance as the original normal distribution
        half_range = np.sqrt(3) * scale
        perturbations = np.random.uniform(loc - half_range, loc + half_range,
                                          size=(num_samples, self.X.shape[1]))
        return perturbations

    # === Step 3: Single experiment ===
    def single_run(self, X_instance):
        r2_group_all = defaultdict(list)
        mse_group_all = defaultdict(list)
        rank_group_all=defaultdict(list)
        use_group_all=defaultdict(list)
        hit_group_all=defaultdict(list)
        d_auc_group_all_p=defaultdict(list)
        d_auc_group_all_d = defaultdict(list)

        kknots, w1, uu_value, kk_set, use_ng,r2_ =  [], [], [], [], [], []
        for m in range(self.N):
            inverse_data = self.generate_perturbations_with_truncnorm(loc=X_instance,num_samples=int(self.n_perturbation/0.8))
            #inverse_data = self.generate_uni_perturbations(loc=X_instance,scale=self.scale,
                                                                    # num_samples=int(self.n_perturbation / 0.8))
            #inverse_data = self.generate_continuous(loc=X_instance,x_train=self.X, num_samples=int(self.n_perturbation/0.8))

            #scaled_data = (inverse_data-self.loc) / (self.scale * self.scale_factor)
            #instance0 = (X_instance.reshape(1, -1)-self.loc) / (self.scale * self.scale_factor)
            denominator = self.scale * self.scale_factor
            # Replace zero denominators with 1 to avoid division by zero; inverse_data / 1 preserves the original values
            denominator = np.where(denominator == 0, 1, denominator)
            scaled_data = (inverse_data-self.loc) / denominator
            instance0 = (X_instance.reshape(1, -1)-self.loc) / denominator

            if self.isscaler:
                predictions = self.model.predict_proba(scaled_data)[:, 0]
            else:
                predictions = self.model.predict_proba(inverse_data)[:, 0]

            distances = pairwise_distances(scaled_data, instance0).ravel()

            kernel_width = np.sqrt(inverse_data.shape[1]) * self.kw
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
            #weights=np.ones(len(weights))

            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                scaled_data, predictions, weights, test_size=0.2, random_state=self.random_state
            )
            continuous_features = range(self.X.shape[1])

            knots = [np.linspace(np.min(X_train[:, idx]), np.max(X_train[:, idx]), self.n0 + 1)
                     for idx in continuous_features]
            #knots = [np.percentile(X_train[:, idx], np.linspace(0, 100, self.n0 + 1))
                     #for idx in continuous_features]

            p_g=np.ones(len(continuous_features))
            #p_g=self.b_global

            results = fit_piecewise_local_model_li(p_g,self.b_global,X_train, y_train, w_train, continuous_features, categorical_idx = [],
            X_test = X_test,  y_test = y_test,weights_test =w_test,
            knots = knots, n0 = self.n0,
            lambda1 = 0.0, lambda_sparse = 0.0)

            col_stds=results["col_stds"]
            uu_value.append(results["u_value"])
            knots=[list(k) for k in knots]
            kknots.append(knots)

            for num in self.groupk:
                steps_k = find_steps_with_exact_k(results["actives"], num, verbose=False)
                #print(steps_k)
                if not steps_k or not steps_k[0]:
                    print(f"[Skip] invalid steps_k: {[steps_k,self.groupk]}")
                    continue  # If inside the for loop

                id_k = steps_k[0][0]

                beta_k=results["beta_hat_path"][id_k]
                M_test_k,y_test_k,w_test_k=results["M_xyw"]
                y_pred_k = M_test_k.dot(beta_k)
                r2_test_k = r2_score(y_test_k, y_pred_k, sample_weight=w_test_k)
                mse_test_k = np.average((y_test_k - y_pred_k) ** 2, weights=w_test_k)
                r2_group_all[num].append(r2_test_k)
                mse_group_all[num].append(mse_test_k)

                kk = [beta_k[i] * col_stds[i] for i in range(self.X.shape[1])]
                sorted_indices_x = np.argsort(kk)[::-1]
                predict_fn = lambda x: self.model.predict_proba(x)[0, 0]
                '''
                d_auc_p = normalized_preservation_auc(predict_fn, X_instance.reshape(1, -1), sorted_indices_x, self.loc)
                d_auc_d=normalized_deletion_auc(predict_fn, X_instance.reshape(1, -1), sorted_indices_x, self.loc)
                d_auc_group_all_p[num].append(d_auc_p)
                d_auc_group_all_d[num].append(d_auc_d)
                '''

                rank_group_all[num].append(kk)

                used_features = steps_k[0][1]
                hit_group_all[num].append(len(set( used_features) & set(range(4))))
                use_group_all[num].append(used_features)

        rank_stab = {k: multiple_run_consistency(v) for k, v in rank_group_all.items()}
        use_stab={k: stability_used_feature(v) for k, v in use_group_all.items()}
        r2_mean = {k: np.mean(v) for k, v in r2_group_all.items()}
        r2_std = {k: np.std(v) for k, v in r2_group_all.items()}
        mse_mean = {k: np.mean(v) for k, v in mse_group_all.items()}
        mse_std = {k: np.std(v) for k, v in mse_group_all.items()}
        hit_mean = {k: np.mean(v) for k, v in hit_group_all.items()}
        rank_mean={k: np.mean(v,axis=0) for k, v in rank_group_all.items()}
        rank_std = {k: np.std(v, axis=0) for k, v in rank_group_all.items()}
        #d_auc_mean_p = {k: np.mean(v, axis=0) for k, v in d_auc_group_all_p.items()}
        #d_auc_mean_d = {k: np.mean(v, axis=0) for k, v in d_auc_group_all_d.items()}


        use_set = defaultdict(list)
        for k,v in use_group_all.items():
            sets = [set(lst) for lst in v]
            common_elements = set.intersection(*sets)
            use_set[k]=common_elements

        return rank_std,rank_mean,rank_group_all,hit_mean, uu_value,kknots,use_set,r2_mean,mse_mean,r2_std,mse_std,rank_stab,use_stab

    def run_analysis(self,run_name=None):
        start_time = time.time()
        X_instance = self.instance
        rank_std,rank_mean,rank_group_all,hit_mean,uu_value, kknots, use_set,r2_mean,mse_mean,r2_std,mse_std,rank_stab,use_stab = self.single_run(X_instance)
        avg_similarity = []
        for i in range(self.X.shape[1]):
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
        mean_slope_similarity = np.mean(avg_similarity)
        end_time = time.time()
        duration = end_time - start_time

        result  = {
            "run_name": run_name or f"run_{len(self.history) + 1}",
            "duration_sec": round(duration, 3),
            "used_features_stability": use_stab,
            "ranking_features_stability": rank_stab,
            "shape_slope_stability": u_,
            "hit_mean":hit_mean,

            "rank_std":rank_std,

            "r2_mean": r2_mean,
            "mse_mean": mse_mean,
            "r2_std": r2_std,
            "mse_std": mse_std,
            "N": self.N,
            #"n_perturbation": self.n_perturbation,
            #"scale_factor": self.scale_factor,
            # "self.instance" : self.instance,
            #"n0": self.n0,
            #"rank_mean": rank_mean,
            "rank_group_all":rank_group_all,
            #"d":d,
            #"kknots": kknots,
            #"uu_value":uu_value,
            #"m_train":m_train[0]

        }
        self.results_summary = result
        self.history.append(result)
        #print(result)

        return result

    def save_results(self):
        df = pd.DataFrame(self.history)
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                df.to_csv(self.save_path, index=False)
            else:
                df.to_csv(self.save_path, mode='a', header=False, index=False)


