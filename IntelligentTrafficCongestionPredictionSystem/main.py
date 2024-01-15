from Data_Preprocess import DataPreprocess
from LightGBM import ModelLightGBM
import argparse
from randomforest import RandomForestModel
from boost_xg import XGBoostModel
from CatBoost import ModelCatBoost

# Rest of the code goes here, based on the selected mode and arguments
parser = argparse.ArgumentParser(description='Train or test LightGBM model')
parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Select mode: train or test')
parser.add_argument('--model', choices=['LightGBM', 'CatBoost', 'XGBoost', 'RandomForest'], default='LightGBM', help='Select model')
# Parse the command-line arguments
args = parser.parse_args()


def main():
    if args.mode == 'train':
        if args.model == 'LightGBM':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            model = ModelLightGBM()
            training_time, train_rmse, train_r2 = model.LightGBM_train_model(X_train, y_train)
            print(f'LightGBM Training time {round(training_time,2)} seconds')
            print(f'LightGBM 5-fold cross validation RMSE {round(train_rmse, 2)}')
            print(f'LightGBM 5-fold cross validation R2 score {round(train_r2, 2)}')
        if args.model == 'XGBoost':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            print('calling model')
            model = XGBoostModel()
            print('Training model')
            training_time, train_rmse, train_r2 = model.XGBoost_train_model(X_train, y_train)
            print(f'XGBoost Training time {round(training_time,2)} seconds')
            print(f'XGBoost 5-fold cross validation RMSE {round(train_rmse, 2)}')
            print(f'XGBoost 5-fold cross validation R2 score {round(train_r2, 2)}')
        if args.model == 'RandomForest':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            model = RandomForestModel()
            training_time, train_rmse, train_r2 = model.random_train_model(X_train, y_train)
            print(f'Random Forest Training time {round(training_time,2)}')
            print(f'Random Forest 5-fold cross validation RMSE {round(train_rmse, 2)}')
            print(f'Random Forest 5-fold cross validation R2 score {round(train_r2, 2)}')
        if args.model == 'CatBoost':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            model = ModelCatBoost()
            training_time, train_rmse, train_r2 = model.catboost_train_model(X_train, y_train)
            print(f'Cat Boost Training time {round(training_time,2)}')
            print(f'Cat Boost 5-fold cross validation RMSE {round(train_rmse, 2)}')
            print(f'Cat Boost 5-fold cross validation R2 score {round(train_r2, 2)}')
    elif args.mode == 'test':
        if args.model == 'LightGBM':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            model = ModelLightGBM()
            test_rmse, test_r2, y_pred = model.LightGBM_test_model(X_test, y_test)
            print('LightGBM Evaluation')
            print(f'RMSE {round(test_rmse, 2)}')
            print(f'R2 score {round(test_r2, 2)}')
        if args.model == 'XGBoost':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            print('calling model')
            model = XGBoostModel()
            print('Training model')
            test_rmse, test_r2, y_pred = model.xgboost_test_model(X_test, y_test)
            print('XGBoost Evaluation')
            print(f'RMSE {round(test_rmse, 2)}')
            print(f'R2 score {round(test_r2, 2)}')
        if args.model == 'RandomForest':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            model = RandomForestModel()
            test_rmse, test_r2, y_pred = model.random_test_model(X_test, y_test)
            print('Random Forest Evaluation')
            print(f'RMSE {round(test_rmse, 2)}')
            print(f'R2 score {round(test_r2, 2)}')
        if args.model == 'CatBoost':
            data_preprocessor = DataPreprocess()
            df = data_preprocessor.load_dataset()
            X_train, y_train, X_test, y_test = data_preprocessor.data_preprocessing()
            model = ModelCatBoost()
            test_rmse, test_r2, y_pred = model.catboost_test_model(X_test, y_test)
            print('Cat Boost Evaluation')
            print(f'RMSE {round(test_rmse, 2)}')
            print(f'R2 score {round(test_r2, 2)}')
    else:
        print('Invalid mode selected. Please choose between train and test.')


if __name__ == "__main__":
    main()
