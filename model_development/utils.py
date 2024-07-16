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


def PCA_feature_selection(X_transformed, max_features=20):
    """
       Feature selection based on PCA loadings.

       Parameters:
       - X_transformed (pd.DataFrame): The input data frame with features.
       - max_features (int): The maximum number of features to select.

       Returns:
       - pd.DataFrame: The data frame with selected features.
       """

    # Perform PCA
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
    final_features = contribution.sort_values(ascending=False)[:max_features].index.to_list()

    # Final X DataFrame containing selected features
    X_selected = X_transformed[final_features]

    return X_selected