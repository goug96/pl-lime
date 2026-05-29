import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
# === Import external functions ===
from new_select_feature import run, compute_stability
from fit_knots import fit_piecewise_local_model1xx_zb
from evaluate import (slope_trend_similarity,multiple_run_consistency,stability_used_feature,deletion_auc_cat1,
                      deletion_auc_r,make_baseline,preservation_auc,normalized_preservation_auc)
import pandas as pd
import os
import time
from data_generate import data_inverse_1
from no_onehot_fit_knots import fit_piecewise_local_model_li
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import sklearn
from new_select_feature import select_features_with_lasso_path
from evaluate import stability_used_feature, multiple_run_consistency
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.linear_model import Ridge
from lime.lime_tabular import LimeTabularExplainer
from collections import defaultdict
class LocalModelStabilityAnalyzer:
    def __init__(self, predict_fn, X, y, instance,categorical_features,continuous_features,groupk=[10], N=50, n_perturbation=6250,scale_factor=1, random_state=42,save_path="results.csv"):
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
    def run_single(self, X_instance):
        r2_group_all = defaultdict(list)
        mse_group_all = defaultdict(list)
        rank_group_all = defaultdict(list)
        use_group_all = defaultdict(list)
        d_auc_group_all = defaultdict(list)
        c_group_all = defaultdict(list)
        for m in range(self.N):
            data1, inverse_data1 = data_inverse_1(self.X, self.instance, self.categorical_features,
                                                  self.continuous_features, self.scale,
                                                  num_samples=int(self.n_perturbation / 0.8))

            predictions = self.predict_fn(inverse_data1)[:, 0]
            #predictions = self.predict_fn(inverse_data1)
            scale = np.ones(len(self.instance))
            scale[self.continuous_features] = self.scale[self.continuous_features]


            data_transformed = data1/scale

            distances = pairwise_distances(data_transformed, data_transformed[0].reshape(1,-1)).ravel()

            kernel_width = np.sqrt(data_transformed.shape[1]) * 0.75
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
            X_train_p, X_test_p, y_train_p, y_test_p, w_train, w_test = train_test_split(
                data_transformed, predictions, weights, test_size=0.2, random_state=self.random_state
            )
            for num in self.groupk:
                used_features = select_features_with_lasso_path(X_train_p, y_train_p, w_train, num)
                model = Ridge(alpha=1, fit_intercept=True)
                model.fit(X_train_p[:, used_features], y_train_p, sample_weight=w_train)
                preds = model.predict(X_test_p[:, used_features])
                r2 = r2_score(y_test_p, preds, sample_weight=w_test)
                mse = np.average((y_test_p - preds) ** 2, weights=w_test)

                col_std = np.std(X_train_p, axis=0)

                use_group_all[num].append(used_features)
                r2_group_all[num].append(r2)
                mse_group_all[num].append(mse)
                coef_abs_full = np.zeros(X_train_p.shape[1])
                coef_abs_full[used_features] = np.abs(model.coef_)
                c_group_all[num].append(model.coef_)

                rank_group_all[num].append(coef_abs_full)
                sorted_indices = np.argsort(coef_abs_full)[::-1]
                baseline = make_baseline(self.X, self.continuous_features, self.categorical_features)
                predict1 = lambda x: self.predict_fn(x)[0, 0]

                d_auc = normalized_preservation_auc(predict1, X_instance.reshape(1, -1), sorted_indices , baseline,
                                         )
                d_auc_group_all[num].append(d_auc)

        rank_stab = {k: multiple_run_consistency(v) for k, v in rank_group_all.items()}
        use_stab = {k: stability_used_feature(v) for k, v in use_group_all.items()}
        r2_mean = {k: np.mean(v) for k, v in r2_group_all.items()}
        r2_std = {k: np.std(v) for k, v in r2_group_all.items()}
        mse_mean = {k: np.mean(v) for k, v in mse_group_all.items()}
        mse_std = {k: np.std(v) for k, v in mse_group_all.items()}
        d_auc_mean = {k: np.mean(v) for k, v in d_auc_group_all.items()}
        #coef_mean= {k: np.mean(v,axis=0) for k, v in c_group_all.items()}

        return c_group_all,d_auc_mean,r2_mean, r2_std, mse_mean, mse_std, rank_stab, use_stab

    # === Step 4: Main workflow ===
    def run_analysis(self, run_name=None):
        start_time = time.time()
        X_instance = self.instance
        c_group_all,d_auc_mean,r2_mean, r2_std, mse_mean, mse_std, rank_stab, use_stab = self.run_single(X_instance)

        end_time = time.time()  #
        duration = end_time - start_time

        result = {
            "run_name": run_name or f"lime_run_{len(self.history) + 1}",
            "duration_sec": round(duration, 3),
            "used_features_stability": use_stab,
            "ranking_features_stability": rank_stab,
            "avg_mse_test": mse_mean,
            "avg_r2_test": r2_mean,
            "avg_mse_std": mse_std,
            "avg_r2_std": r2_std,
            "N": self.N,
            #"coef_mean":coef_mean,
            "c_group_all": c_group_all,
            "d_auc_mean":d_auc_mean,
            "n_perturbation": self.n_perturbation,

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



if __name__ == "__main__":
    '''
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    df.to_csv("adult.csv", index=False)

    # 2. Drop missing values
    df = df.dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = df.drop(columns='income').columns.tolist()
    labels_raw = df['income'].values
    data_raw = df.drop(columns='income').values.astype(str)  # Convert all values to strings for LabelEncoder

    # 2. Label encoding
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    class_names = le.classes_

    # 3. Determine categorical variable indices (assuming non-numeric columns are categorical)
    categorical_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    ]
    n_features=len(feature_names)
    feature_names = df.drop(columns='income').columns.tolist()
    categorical_features = [feature_names.index(col) for col in categorical_columns]
    continuous_features = [i for i in range(n_features) if i not in categorical_features]

    categorical_names = {}
    for feature in categorical_features:
        le_feat = LabelEncoder()
        le_feat.fit(data_raw[:, feature])
        data_raw[:, feature] = le_feat.transform(data_raw[:, feature])
        categorical_names[feature] = le_feat.classes_

    data = data_raw.astype(float)
    variances = np.zeros(data.shape[1])
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # categorical_features is already a list of indices
    # Create ColumnTransformer: pass through numeric columns and one-hot encode categorical columns
    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    train, test, labels_train, labels_test = train_test_split(data, labels, train_size=0.80, stratify=labels,random_state=42)
    encoder.fit(data)
    encoded_train = encoder.transform(train)
    encoded_test = encoder.transform(test)

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(encoded_train, labels_train)

    # Test-set accuracy
    acc = sklearn.metrics.accuracy_score(labels_test, clf.predict(encoded_test))
    print("Accuracy:", acc)
    import joblib
    import numpy as np

    import joblib

    # Save the trained model
    joblib.dump(clf, "rf_model.pkl")

    # Save the preprocessor (OneHotEncoder)
    joblib.dump(encoder, "encoder.pkl")
    '''
    '''
    df = pd.read_csv("adult.csv")

    df = df.dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = df.drop(columns='income').columns.tolist()
    labels_raw = df['income'].values
    data_raw = df.drop(columns='income').values.astype(str)  # Convert all values to strings for LabelEncoder

    # 2. Label encoding
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    class_names = le.classes_

    # 3. Determine categorical variable indices (assuming non-numeric columns are categorical)
    categorical_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    ]
    n_features = len(feature_names)
    feature_names = df.drop(columns='income').columns.tolist()
    categorical_features = [feature_names.index(col) for col in categorical_columns]
    continuous_features = [i for i in range(n_features) if i not in categorical_features]

    categorical_names = {}
    for feature in categorical_features:
        le_feat = LabelEncoder()
        le_feat.fit(data_raw[:, feature])
        data_raw[:, feature] = le_feat.transform(data_raw[:, feature])
        categorical_names[feature] = le_feat.classes_

    data = data_raw.astype(float)
    variances = np.zeros(data.shape[1])
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    train, test, labels_train, labels_test = train_test_split(data, labels, train_size=0.80, stratify=labels,
                                                              random_state=42)
    encoder.fit(data)
    encoded_train = encoder.transform(train)
    encoded_test = encoder.transform(test)

    import joblib

    clf = joblib.load("rf_model.pkl")
    encoder = joblib.load("encoder.pkl")
    instance = test[0]
    acc = sklearn.metrics.accuracy_score(labels_test, clf.predict(encoded_test))
    print("Accuracy:", acc)

    # 7. LIME explainer
    predict_fn = lambda x: clf.predict_proba(encoder.transform(np.array(x)))
    instance= test[0]
    explainer = LimeTabularExplainer(
        training_data=train,
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        discretize_continuous=False,
        sample_around_instance=True,
        feature_selection='lasso_path'
    )

    predict_fn = lambda x: clf.predict_proba(encoder.transform(x)).astype(float)
    analyzer = LocalModelStabilityAnalyzer(predict_fn,train, labels_train,instance, categorical_features=categorical_features,continuous_features=continuous_features,k1=20, N=50,n_perturbation=5000,n0=1,
                                           save_path="a_lime_results.csv")

    for k in [6,8,10,12,14]:
        analyzer.k1 = k
        analyzer.run_analysis(run_name=f"save_number{k}")

    analyzer.save_results()

    #analyzer.save_results("runs_results.csv")
    '''
