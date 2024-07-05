class ModelTraining:
    """
       Class for training and evaluating a CatBoostRegressor model with optional feature selection and hyperparameter tuning.

       Parameters:
       -----------
       feature_selection (bool): Whether to perform feature selection before model training.
       grid_search_params (dict or None): Parameters for grid search cross-validation. If None, default parameters are used.
       feature_selection_method {'from_model', 'pca'}: Method for feature selection:
           - 'from_model': Uses SelectFromModel with CatBoostRegressor for feature selection.
           - 'pca': Uses PCA for feature selection.

       Attributes:
       -----------
       X_selected (pd.DataFrame): Selected features after feature selection.
       y (pd.Series): Target variable.
       selected_features (list): Names of selected features in case feature_selection=True
       grid_search_result (pd.DataFrame): Results of grid search cross-validation, including hyperparameters and evaluation metrics.
       best_params_ (dict): Best hyperparameters found during the grid search.
       model(CatBoostRegressor): Trained CatBoostRegressor model with best_params_ parameters.

       Methods:
       --------
       fit(X, y, max_features=20):
           Performs feature selection (if enabled), hyperparameter tuning, and fits the model to the data using best parameters during grid search.
       predict(X_test=None):
           Predicts the train data by default using the trained model if test data is not provided.
       get_train_test_performance():
           Computes and prints performance metrics (MSE, MAE, R2) on the train/test split of the training data.
       set_final_model(rank):
           Resets the desired model found in grid search results, other than the best model fitted by default.
    """


    def __init__(self, feature_selection=True, grid_search_params=None, feature_selection_method='from_model', cv=5):
        self.feature_selection = feature_selection
        self.grid_search_params = grid_search_params
        self.feature_selection_method = feature_selection_method
        self.cv = cv

        if self.grid_search_params is None:
            self.grid_search_params = {}

    def fit(self, X, y, max_features=20):
        """
            Performs feature selection (if enabled), hyperparameter tuning, and fits the model to the data using best parameters during grid search.

            Split the data into test and train

            Parameters:
            - X (pd.DataFrame): The input data frame with features.
            - y (pd.Series): Target variable.
            - max_feaures (int): Max features allowed to be selected during feature selection.

            Returns:
            None: This method does not return any value while it sets the following attributes:
            - self.X_selected (pd.DataFrame): The selected features after feature selection.
            - self.y (pd.Series): The target variable.
            - self.selected_features (Index): Selected feature names.
            - self.grid_search_result (pd.DataFrame): The result of the grid search.
            - self.best_params_ (dict): The best parameter set found by the grid search.
            - self.model (CatBoostRegressor): The trained model with the best parameter set.
        """
        if self.feature_selection:
            print("Feature Selection has been started")

            if self.feature_selection_method == 'from_model':
                self.selector = SelectFromModel(CatBoostRegressor(verbose=False), max_features=max_features)
                self.selector.fit(X, y)
                X_selected = self.selector.transform(X)
            elif self.feature_selection_method == 'pca':
                X_selected = PCA_feature_selection(X)
        else:
            X_selected = X

        self.X_selected = X_selected
        self.y = y
        self.selected_features = X_selected.columns

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

        # Set the best parameters
        print("\nBest model is being fitted..")
        self.best_params_ = grid_search.best_params_
        self.model = CatBoostRegressor(**self.best_params_, verbose=False)
        self.model.fit(X_selected, y)

    def predict(self, X_test=None):
        """
            Predicts the train data by default using the trained model if test data is not provided.

            Parameters:
            - X_test (pd.DataFrame): Test data frame, default is None, it returns train data predictions.

            Returns:
            -

        :return:
        """
        if X_test is None:
            return self.model.predict(self.X_selected)
        elif self.feature_selection is None:
            return self.model.predict(X_test)
        else:
            X_test_selected = self.selector.transform(X_test)
            return self.model.predict(X_test_selected)

    def get_train_test_performance(self):

        X_selected_train, X_selected_test, y_train, y_test = train_test_ind(self.X_selected, self.y)

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

        print(
            f'MSE Train: {mse_train:.4f} \nMSE Test: {mse_test:.4f} \n \nMAE Train: {mae_train:.4f} \nMAE Test: {mae_train:.4f} \n \nR2 Train: {r2_train:.4f} \nR2 Test: {r2_test:.4f}')

    def set_final_model(self, rank):

        df_best_params = self.grid_search_result.query("rank_test_score==@rank").filter(like='param')
        df_best_params.columns = df_best_params.columns.str.replace('param_', '')
        self.best_params_ = df_best_params.iloc[0].to_dict()

        self.model = CatBoostRegressor(**self.best_params_, verbose=False)
        self.model.fit(self.X_selected, self.y)