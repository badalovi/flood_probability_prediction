import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from .utils import *


class ModelTraining:
    """
       Class for training and evaluating a CatBoostRegressor model with optional feature selection and hyperparameter tuning.

       Parameters:
       - feature_selection (bool): Whether to perform feature selection before model training.
       - grid_search_params (dict or None): Parameters for grid search cross-validation. If None, default parameters are used.
       - feature_selection_method {'from_model', 'pca'}: Method for feature selection:
            - 'from_model': Uses SelectFromModel with CatBoostRegressor for feature selection.
            - 'pca': Uses PCA Loadings for feature selection.
       - cv (int): Number of folds in Cross-Validation

       Attributes:
       - X_selected (pd.DataFrame): Selected features after feature selection.
       - y (pd.Series): Target variable.
       - selected_features (list): Names of selected features in case feature_selection=True
       - grid_search_result (pd.DataFrame): Results of grid search cross-validation, including hyperparameters and evaluation metrics.
       - best_params_ (dict): Best hyperparameters found during the grid search.
       - model(CatBoostRegressor): Trained CatBoostRegressor model with best_params_ parameters.

       Methods:
       - fit(X, y, max_features=20):
            Performs feature selection (if enabled), hyperparameter tuning, and fits the model to the data using best parameters during grid search.
       - predict(X_test=None):
            Predicts the train data by default using the trained model if test data is not provided.
       - get_train_test_performance():
            Computes and prints performance metrics (MSE, MAE, R2) on the train/test split of the training data.
       - set_final_model(rank):
            Resets the desired model found in grid search results, other than the best model fitted by default.
    """


    def __init__(self, feature_selection=True, grid_search_params=None, feature_selection_method='from_model', cv=5):
        self.feature_selection = feature_selection
        self.grid_search_params = grid_search_params
        self.feature_selection_method = feature_selection_method
        self.cv = cv
        self.model_train = None

        if self.grid_search_params is None:
            self.grid_search_params = {}

    def fit(self, X, y, max_features=20):
        """
            Performs feature selection (if enabled), hyperparameter tuning, and fits the model to the data using best parameters found during grid search.

            Parameters:
            - X (pd.DataFrame): The input data frame with features.
            - y (pd.Series): Target variable.
            - max_features (int): Max features allowed to be selected during feature selection.

            Returns:
            None: This method does not return any value while it sets the following attributes:
            - self.X_selected (pd.DataFrame): The selected features after feature selection.
            - self.y (pd.Series): The target variable.
            - self.selected_features (Index): Selected feature names.
            - self.grid_search_result (pd.DataFrame): The result of the grid search.
            - self.best_params_ (dict): The best parameter set found by the grid search.
            - self.model (CatBoostRegressor): The trained model with the best parameter set.
        """

        ### 1. Feature Selection Step
        if self.feature_selection:
            print("Feature Selection has been started")

            if self.feature_selection_method == 'from_model':
                self.selector = SelectFromModel(CatBoostRegressor(verbose=False), max_features=max_features)
                self.selector.fit(X, y)
                X_selected = self.selector.transform(X)
            elif self.feature_selection_method == 'pca':
                self.selector = pca_selector(max_features=max_features)
                self.selector.fit(X)
                X_selected = self.selector.transform(X)
        else:
            X_selected = X

        self.X_selected = X_selected
        self.y = y
        self.selected_features = X_selected.columns


        ### 2. Hyperparameter Tuning Step
        print("\nHyperparameter Tuning has been started")

        grid_search = GridSearchCV(
            CatBoostRegressor(verbose=False),
            self.grid_search_params,
            scoring='neg_mean_squared_error',
            verbose=250,
            n_jobs=-1,
            cv=self.cv
        )

        grid_search.fit(self.X_selected, self.y)
        self.grid_search_result = pd.DataFrame(grid_search.cv_results_) \
                                      .loc[:, ['param_depth', 'param_iterations', 'param_learning_rate',
                                               'mean_test_score', 'std_test_score', 'rank_test_score']]

        print("\nHyperparameter tuning has been done")


        ### 3. Final Model Fitting Step
        print("\nBest model is being fitted..")
        self.best_params_ = grid_search.best_params_
        self.model = CatBoostRegressor(**self.best_params_, verbose=False)
        self.model.fit(X_selected, y)

        # Initializing rank parameters to avoid repetition
        self.rank_current = 0
        self.rank_input = 1

    def predict(self, X_test=None):
        """
            Predicts the train data by default using the trained model if test data is not provided.

            Parameters:
            - X_test (pd.DataFrame): Test data frame, default is None, it returns train data predictions.

            Returns:
            - Returns predictions for train sample by default or predictions for test sample if X_test is provided
        """
        if X_test is None:
            return self.model.predict(self.X_selected)
        elif self.feature_selection is None:
            return self.model.predict(X_test)
        else:
            X_test_selected = self.selector.transform(X_test)
            return self.model.predict(X_test_selected)

    def get_train_test_performance(self):
        """
            Prints train and test performance metrics by splitting data, and fitting a model on train data
            using parameters in self.best_params_.

            This method is especially developed to benchmark the tuned model with a simple model
            which has no any further modifications.

            Parameters:
            - None.

            Returns:
            - None
        """

        # Split the main training data
        X_selected_train, X_selected_test, y_train, y_test = train_test_ind(self.X_selected, self.y)

        # Check if the fitted model is already fitted
        if self.rank_current != self.rank_input:
            self.rank_current = self.rank_input
            self.model_train = CatBoostRegressor(**self.best_params_, verbose=False)
            self.model_train.fit(X_selected_train, y_train)

        y_pred_train = self.model_train.predict(X_selected_train)
        y_pred_test = self.model_train.predict(X_selected_test)

        mse_train = mean_squared_error(y_pred_train, y_train)
        mse_test = mean_squared_error(y_pred_test, y_test)

        mae_train = mean_absolute_error(y_pred_train, y_train)
        mae_test = mean_absolute_error(y_pred_test, y_test)

        r2_train = r2_score(y_pred_train, y_train)
        r2_test = r2_score(y_pred_test, y_test)

        print(f'MSE Train: {mse_train:.4f} \nMSE Test: {mse_test:.4f} \n \nMAE Train: {mae_train:.4f} \nMAE Test: {mae_test:.4f} \n \nR2 Train: {r2_train:.4f} \nR2 Test: {r2_test:.4f}')

    def set_final_model(self, rank):
        """
            Refit the final model based on the given model rank.

            Parameters:
            - rank (int): Rank of the model which is provided in the self.grid_search_result

            Returns:
            - None: This method does not return any value while it resets the following attributes:
            - self.best_params_ (dict): The best parameter set chosen by user from the self.grid_search_result.
            - self.model (CatBoostRegressor): The trained model with the best_params_ parameter set.
        """
        # Check if the input model is already fitted
        if self.rank_input != rank:
            self.rank_input = rank

            df_best_params = self.grid_search_result.query("rank_test_score==@rank").filter(like='param')
            df_best_params.columns = df_best_params.columns.str.replace('param_', '')
            self.best_params_ = df_best_params.iloc[0].to_dict()
            self.model = CatBoostRegressor(**self.best_params_, verbose=False)
            self.model.fit(self.X_selected, self.y)