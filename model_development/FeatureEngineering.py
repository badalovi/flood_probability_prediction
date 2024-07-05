class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
       Custom transformer for feature engineering.

       Parameters:
       -----------
       keep_original_features (bool): Flag indicating whether to keep original features in the transformed data.
    """

    def __init__(self, keep_original_features=True):
        self.keep_original_features = keep_original_features

    def fit(self, X):

        # There is nothing to fit for feature engineering in this project
        # It is still added as a method to be compatible with sklearn fashion
        pass

    def transform(self, X):
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

