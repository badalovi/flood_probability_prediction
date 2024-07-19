import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def train_test_ind(X, y, seed_=100):
    """
        Split the data into test and train

        Parameters:
        - X (pd.DataFrame): The input data frame with features.
        - y (pd.Series): Target variable.
        - seed_ (int): Random seed for reproducibility

        Returns:
        tuple: A tuple containing the DataFrames and Series:
        - X_train (pd.DataFrame): The training set features.
        - X_test (pd.DataFrame): The test set features.
        - y_train (pd.Series): The training set target variable.
        - y_test (pd.Series): The test set target variable.
    """

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        X = np.array(X)
        y = np.array(y)

    # Setting seed for reusability
    np.random.seed(seed_)

    # Set test size
    test_size = int(0.3 * len(X))

    # Create permutation of indices
    indices = np.random.permutation(X.shape[0])

    train_ind = indices[test_size:]
    test_ind = indices[:test_size]

    X_train = X[train_ind]
    X_test = X[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]

    return X_train, X_test, y_train, y_test

class pca_selector():
    """
           Customer class for PCA feature selection.

           Parameters:
           max_features (int): The maximum number of features to select.

           Attributes:
           max_features (int): The maximum number of features to select.

           Methods:
           fit(self, X):
                Fit method, does nothing as no fitting is required for this transformer.
           transform(self, X):
                Transform original features
        """

    def __init__(self, max_features=20):
        self.max_features = max_features

    def fit(self, X_transformed):
        """
            This method performs feature selection using Principal Component Analysis (PCA) loadings and save
            the selected features as self.selected_features.

            Parameters:
            - X_transformed (pd.DataFrame): The input data frame with features.

            Returns:
            This method does not return anything while saving selected features as self.selected_features
            to pass to transform() method.
        """

        pca = PCA()
        pca.fit(X_transformed)

        # Get the explained variance ratio and calculate loadings
        explained_variance = pca.explained_variance_ratio_

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_df = pd.DataFrame(loadings, index=X_transformed.columns,
                                  columns=[f'PC_{i + 1}' for i in range(len(X_transformed.columns))])

        # Calculate the contribution of each feature to the total explained variance
        contribution = loading_df.abs().sum(axis=1)
        contribution = contribution / contribution.sum()

        # Set final features
        self.final_features = contribution.sort_values(ascending=False)[:self.max_features].index.to_list()

    def transform(self, X_transformed):
        """
        This method selects features selected based on the PCA loadings and returns the
        final DataFrame containing only the final features.

        Parameters:
        - X_transformed (pd.DataFrame): The input data frame with features.

        Returns:
        - pd.DataFrame: The dataframe with selected features
        """
        return X_transformed[self.final_features]

    def fit_transform(self, X_transformed):
        """
        This method performs feature selection using Principal Component Analysis (PCA) loadings and
        returns the final DataFrame containing only the final features

        Parameters:
        - X_transformed (pd.DataFrame): The input data frame with features.

        Returns:
        - pd.DataFrame: The dataframe with selected features
        """
        self.fit(X_transformed)

        return self.transform(X_transformed)