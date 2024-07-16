class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
       Custom transformer for feature engineering.

       Parameters:
       keep_original_features (bool): Flag indicating whether to keep original features in the transformed data.

       Attributes:
       keep_original_features (bool): The flag indicating whether to keep original features in the transformed data.

       Methods:
       fit(self, X):
            Fit method, does nothing as no fitting is required for this transformer.
       transform(self, X):
            Transform original features

    """

    def __init__(self, keep_original_features=True):
        self.keep_original_features = keep_original_features

    def fit(self, X):
        """
            Fit method, does nothing as no fitting is required for this transformer.
            It is still added as a method to be compatible with sklearn fashion.

            Parameters:
            - X (pd.DataFrame): The input data frame with features.

            Returns:
            - self (object): Returns self.
        """

        return self

    def transform(self, X):
        """
            Applies a specified feature engineering approach to original features

            Parameters:
            - X (pd.DataFrame): The input data frame with features.

            Returns:
            - pd.DataFrame: The dataframe with transformed features
        """
        X_transformed = X.copy()

        X = X.assign(
            feat_sum=X.sum(axis=1),
            feat_mean=X.mean(axis=1),
            feat_median=X.median(axis=1),
            feat_var=X.var(axis=1),
            feat_quantile_25=X.quantile(0.25, axis=1),
            feat_quantile_75=X.quantile(0.75, axis=1),
            feat_quantile_90=X.quantile(0.90, axis=1),
            feat_max=X.max(axis=1),
            feat_min=X.min(axis=1),
            feat_rng=X.max(axis=1) - X.min(axis=1),
            feat_krt=X.apply(lambda row: kurtosis(row), axis=1),
            feat_skw=X.apply(lambda row: skew(row), axis=1)
        )

        if not self.keep_original_features:
            X_transformed.drop(X.columns, axis=1)
        return X_transformed

