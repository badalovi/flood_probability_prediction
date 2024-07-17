## Flood Probability Prediction
### Developing a simple ML model training pipeline

This repository demonstrates a simple machine learning workflow for a specific kaggle competition problem, including feature preparation, feature selection, 
hyperparameter tuning, model fitting, and prediction.
<br>It is designed as a showcase to illustrate how to develop a proper machine learning workflow using Python and OOP properly while the goal is not to achieve the best possible model.
<br><br>The workflow is assuming a pre-determined feature engineering approach and a specific ML algorithm(CatBoost).
<br><br>Additionally, using the following methods of *ModelTraining* module a user:
 - *get_train_test_performance* Cang get performance metrics based on train test split, rather than cv results within grid_search_result attribute
 - *set_final_model* Can choose the final model manually from grid_search_result attribute, by providing rank.

<br>
<img width="450" alt="wflowp1" src="https://github.com/user-attachments/assets/6991b610-45b6-48de-b2d5-d27f8fc9c2d8">
