from scipy.stats import truncnorm
import numpy as np
import collections


def calculate_categorical_frequencies(data, categorical_features):

    # Initialize storage structures keyed by global indices
    feature_values = {}  # {global_index: [possible values]}
    feature_frequencies = {}  # {global_index: [corresponding value frequencies]}

    # Iterate over all categorical features using global indices
    for global_idx in categorical_features:
        # Extract this feature column using the global index
        column = data[:, global_idx]

        # Count frequencies
        counter = collections.Counter(column)
        sorted_items = sorted(counter.items())

        # Separate values and counts
        values, counts = map(list, zip(*sorted_items)) if sorted_items else ([], [])

        # Compute frequencies
        total = sum(counts)
        frequencies = [count / total for count in counts] if total > 0 else []

        # Store results by global index
        feature_values[global_idx] = values
        feature_frequencies[global_idx] = frequencies

    return {
        'feature_values': feature_values,
        'feature_frequencies': feature_frequencies
    }
def generate_perturbations_with_truncnorm(loc,scale,num_samples=10000,scale_factor=1, trunc_a=-2, trunc_b=2):
    d = len(scale)
    perturbations = np.zeros((num_samples, d))

    for i in range(d):
        #mean_i = loc[0, i]  # Extract a single scalar
        mean_i = loc[i]
        std_i = scale[i] * scale_factor
        if std_i == 0:
            std_i = 1e-6

        a, b = trunc_a, trunc_b
        samples = truncnorm.rvs(a, b, loc=mean_i, scale=std_i, size=num_samples)
        perturbations[:, i] = samples

    return perturbations

def generate_continuous_perturbations(loc,scale,num_samples=10000):
    noise=np.random.normal(0, scale=scale, size=(num_samples, len(scale)))
    perturbations = loc + noise
    return perturbations

def generate_continuous(loc, x_train, num_samples=2000):
        # Compute the covariance matrix of x_train
        cov_matrix = np.cov(x_train, rowvar=False)

        # Generate perturbation samples from the covariance matrix
        noise = np.random.multivariate_normal(np.zeros(x_train.shape[1]), cov_matrix, size=num_samples)

        # Add the location offset loc
        perturbations = loc + noise
        return perturbations

def data_inverse_1(data,instance,categorical_features,continuous_features, std,num_samples=5000):
    rng = np.random
    total_features = len(categorical_features) + len(continuous_features)

    data1 = np.zeros((num_samples, total_features))
    inverse_data1 = np.zeros((num_samples, total_features))
    inc = instance.reshape(1, -1)[:, continuous_features]
    # Process continuous features

    X_grid = generate_perturbations_with_truncnorm(instance[continuous_features], std[continuous_features],
                                                   num_samples=num_samples, scale_factor=1, trunc_a=-2, trunc_b=2)
    # X_grid = generate_continuous_perturbations(instance[continuous_features],std[continuous_features],num_samples=num_samples)
    X_grid[0] = instance[continuous_features]

    # Process categorical features and fill them into their original positions
    feature = calculate_categorical_frequencies(data, categorical_features)
    feature_frequencies = feature['feature_frequencies']
    feature_values = feature['feature_values']
    for nb, col_idx in enumerate(categorical_features):
        values = feature_values[col_idx]
        freqs = feature_frequencies[col_idx]
        # Sample categorical feature values
        inverse_column = rng.choice(values, size=num_samples, p=freqs)
        binary_column = (inverse_column == instance[col_idx]).astype(int)

        data1[:, col_idx] = binary_column
        inverse_data1[:, col_idx] = inverse_column

    for nb, col_idx in enumerate(continuous_features):
        data1[:, col_idx] = X_grid[:, nb]
        inverse_data1[:, col_idx] = X_grid[:, nb]

    data1[0, categorical_features] = np.ones(len(categorical_features))
    inverse_data1[0] = instance  # Use the original instance values directly

    return data1, inverse_data1

def data_inverse_co(data,instance,categorical_features,continuous_features, std,num_samples=5000):
    rng = np.random
    total_features = len(categorical_features) + len(continuous_features)

    data1 = np.zeros((num_samples, total_features))
    inverse_data1 = np.zeros((num_samples, total_features))
    inc = instance.reshape(1, -1)[:, continuous_features]
    # Process continuous features

    X_grid = generate_perturbations_with_truncnorm(instance[continuous_features], std[continuous_features],
                                                   num_samples=num_samples, scale_factor=1, trunc_a=-2, trunc_b=2)
    # X_grid = generate_continuous_perturbations(instance[continuous_features],std[continuous_features],num_samples=num_samples)
    X_grid = generate_continuous(instance[continuous_features], data[:,continuous_features], num_samples=num_samples)
    X_grid[0] = instance[continuous_features]

    # Process categorical features and fill them into their original positions
    feature = calculate_categorical_frequencies(data, categorical_features)
    feature_frequencies = feature['feature_frequencies']
    feature_values = feature['feature_values']
    for nb, col_idx in enumerate(categorical_features):
        values = feature_values[col_idx]
        freqs = feature_frequencies[col_idx]
        # Sample categorical feature values
        inverse_column = rng.choice(values, size=num_samples, p=freqs)
        binary_column = (inverse_column == instance[col_idx]).astype(int)

        data1[:, col_idx] = binary_column
        inverse_data1[:, col_idx] = inverse_column

    for nb, col_idx in enumerate(continuous_features):
        data1[:, col_idx] = X_grid[:, nb]
        inverse_data1[:, col_idx] = X_grid[:, nb]

    data1[0, categorical_features] = np.ones(len(categorical_features))
    inverse_data1[0] = instance  # Use the original instance values directly

    return data1, inverse_data1
