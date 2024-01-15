import time
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score


class ModelLightGBM:
    def __init__(self):
        # Set the necessary parameters for the LightGBM regressor

        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 512,
            'max_depth': 9,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'num_threads': 4,
            'n_estimators': 150,
            'min_child_weight': 3
        }

    def LightGBM_train_model(self, X_train, y_train):

        """
        Train a LightGBM model using k-fold cross-validation and evaluate its performance.

        Params:
            - X_train (DataFrame): The input features for training.
            - y_train (DataFrame): The target variables for training.

        Returns:
            - training_time (float): The total training time in seconds.
            - overall_rmse (float): The overall root mean squared error (RMSE) across all target variables.
            - overall_r2 (float): The overall R-squared score across all target variables.
        """
        # Initialize LightGBM regressor
        model = lgb.LGBMRegressor(**self.params)

        # Initialize MultiOutputRegressor with LightGBM model
        multi_output_model = MultiOutputRegressor(model)

        # Perform k-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []
        fold_no = 1
        total_training_time = 0

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            # Start the training time measurement
            start_time = time.time()

            # Train the model
            model_lightGBM = multi_output_model.fit(X_train_fold, y_train_fold)

            # End the training time measurement
            end_time = time.time()

            # Calculate the training time
            training_time = end_time - start_time
            print(f"Training time for fold {fold_no - 1}: {round(training_time, 2)} seconds")

            # Accumulate the training time for all folds
            total_training_time += training_time

            # Make predictions on the validation set
            y_pred = model_lightGBM.predict(X_val)

            # Calculate RMSE for each target variable
            rmse_fold = [mean_squared_error(y_val.iloc[:, i], y_pred[:, i], squared=False) for i in
                         range(y_val.shape[1])]
            rmse_scores.append(rmse_fold)

            # Calculate R-squared score for each target variable
            r2_fold = [r2_score(y_val.iloc[:, i], y_pred[:, i]) for i in range(y_val.shape[1])]
            r2_scores.append(r2_fold)


        # Calculate overall RMSE and R-squared score
        rmse_scores = np.mean(rmse_scores, axis=0)
        r2_scores = np.mean(r2_scores, axis=0)
        print(f'rmse_scores = {rmse_scores}')
        print(f'r2_scores = {r2_scores}')

        # Print RMSE and R-squared score for each target variable
        for i, target_col in enumerate(y_train.columns):
            print(f'RMSE for {target_col}: {round(rmse_scores[i], 2)}')
            print(f'R-squared for {target_col}: {round(r2_scores[i], 2)}')

        # Step 10: Calculate overall RMSE
        overall_rmse = np.mean(rmse_scores)
        print(f'Overall RMSE: {round(overall_rmse, 2)}')

        overall_r2 = np.mean(r2_scores)
        print(f'Overall r2: {round(overall_r2, 2)}')

        # # Step 1: Import the necessary library
        # import pickle

        # Step 2: Save the trained model after the K-fold training loop
        with open('kfold_LightGBM_model.pkl', 'wb') as file:
            pickle.dump(model_lightGBM, file)

        # # Step 3: Load the saved model and predict on X_test
        # with open('kfold_model.pkl', 'rb') as file:
        #     loaded_model = pickle.load(file)

        # Make predictions on X_test using the loaded model
        # y_pred = loaded_model.predict(X_test)

        # Get feature importances for each individual regressor
        feature_importances = []
        for regressor in multi_output_model.estimators_:
            feature_importances.append(regressor.feature_importances_)

        # Average feature importances across all regressors
        average_importances = pd.DataFrame(feature_importances).mean().values

        # Convert feature names to strings
        feature_names = [str(col) for col in X_train.columns]

        # Sort feature importances in descending order
        sorted_indices = np.argsort(average_importances)[::-1]
        sorted_importances = average_importances[sorted_indices]
        sorted_feature_names = np.array(feature_names)[sorted_indices]

        # Visualize feature importance
        plt.figure(figsize=(6, 6))
        plt.barh(sorted_feature_names, sorted_importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('LightGBM - Feature Importance', fontweight='bold')
        plt.show()
        return total_training_time, overall_rmse, overall_r2

    def LightGBM_test_model(self, X_test, y_test):
        """
           Make predictions on the test data using the loaded model.

           Parameters:
           - X_test (array-like or DataFrame): Test data features.

           Returns:
           - array-like: Predicted values.
        """
        with open('kfold_LightGBM_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        # Make predictions on X_test using the loaded model
        y_pred = loaded_model.predict(X_test)

        # Calculate RMSE
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Calculate R2 score
        test_r2 = r2_score(y_test, y_pred)

        return test_rmse, test_r2, y_pred

