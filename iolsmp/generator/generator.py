import numpy as np
import pandas as pd

def generate_clustered_data(num_clusters, num_observations, num_features, error_expectation = 1):
    """
    Generate synthetic clustered data with specified parameters.

    Parameters:
        num_clusters (int): Number of clusters.
        num_observations (int): Number of observations.
        error_expectation (float): Expectation of the error term.
        num_features (int): Number of independent variables (features).

    Returns:
        pd.DataFrame: DataFrame containing the generated clustered data with columns 'X1', 'X2', 'Cluster', and 'Y'.

    """

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    # Independent variables
    X = np.random.uniform(1, 2, size=(num_observations, num_features))

    # Cluster variable
    clusters = np.repeat(range(num_clusters), num_observations // num_clusters)

    # Generate dependent variable with clustering effect
    beta_true = np.random.uniform(size=num_features)  # True coefficients
    u = 0.5 * np.random.normal(size=num_observations) + \
        0.2 * np.random.normal(size=num_clusters)[clusters]
    nu = error_expectation + u / np.exp(np.dot(X, beta_true))
    Y = np.exp(np.dot(X, beta_true)) * nu

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f'X{i}' for i in range(1, num_features + 1)])
    data['Cluster'] = clusters
    data['Y'] = Y

    return data
