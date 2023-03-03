__author__ = "Soumyadip Majumder"
__version__ = "1.0.7"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "25 Feb 2023"

import pandas as pd
import numpy as np
import joblib
import json
import gc
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from utils.preprocessing import Preprocessor


class HyperOpt:

    def __init__(self) -> None:
        pass

    def ridge_hyper_tuned(self, X_train, y_train, X_val, y_val, response_item:str):
        """Tune the hyper parameters of Ridge Regression
        Choose the best params based on Validation Data
        Save the best model based on Validation Data
        
        input: Dict, array, str
        response: binary file"""


        self.alphas = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.copy_xs = [True]
        self.max_iters = range(1000, 10000, 1000)
        self.tols = [1e-2, 1e-3, 1e-4]

        self.performance_df = pd.DataFrame()

        for self.alpha in self.alphas:
            for self.max_iter in self.max_iters:
                for self.tol in self.tols:
                    for self.copy_x in self.copy_xs:
                        self.model = Ridge(alpha=self.alpha, 
                                            copy_X=self.copy_x,
                                            max_iter=self.max_iter,
                                            tol=self.tol,
                                            random_state=143)
                        self.model.fit(X_train, y_train)
                        self.pred_val = self.model.predict(X_val)
                        self.rmse = np.sqrt(mean_squared_error(y_val, self.pred_val, squared=True))
                        self.performance_metrics = {"alpha": self.alpha, 
                                                "copy_X": self.copy_x,
                                                "max_iter": self.max_iter,
                                                "tol": self.tol,
                                                "rmse": self.rmse}
                        print(self.performance_metrics)
    

                        self.df_dict = pd.DataFrame([self.performance_metrics])
                        self.performance_df = pd.concat([self.performance_df, self.df_dict], ignore_index=True)
        self.performance_df.to_excel(f"./outputs/{response_item}_ridge_hyperparams.xlsx", index=False)
        self.best_params = self.performance_df.sort_values(by="rmse", ascending=True).head(1).to_dict(orient="records")[0]

        self.best_params_fit = self.best_params.copy()
        del self.best_params_fit["rmse"]
        self.best_model = Ridge(**self.best_params_fit)
        self.best_model.fit(X_train, y_train)
        
        with open(f"./models/{response_item}_ridge_hypertuned.bin", "wb") as f_out:
             joblib.dump(self.best_model, f_out, compress=3)

        print(f"{response_item} Hyperparameter Tune Ridge Model available")
        
        del self.performance_df, self.best_params_fit
        gc.collect()
        return {"model": self.best_model.__class__.__name__, "rmse": self.best_params["rmse"]}
    
    def randomforest_hyper_tuned(self, X_train, y_train, X_val, y_val, response_item:str):
        """Tune the hyper parameters of Random Forest Regression
        Choose the best params based on Validation Data
        Save the best model based on Validation Data
        
        input: Dict, array, str
        response: dict, binary file"""


        self.n_estimators = range(10, 150, 10)
        self.criterions = ["squared_error", "friedman_mse"]
        self.oob_scores = [True]
        
        self.performance_df = pd.DataFrame()

        for self.n_estimator in self.n_estimators:
            for self.criterion in self.criterions:
                for self.oob_score in self.oob_scores:

                    self.model = RandomForestRegressor(n_estimators=self.n_estimator,
                                                    criterion=self.criterion,
                                                    oob_score=self.oob_score,
                                                    n_jobs=-1,
                                                    verbose=2,
                                                    random_state=143)
                    self.model.fit(X_train, y_train)
                    self.pred_val = self.model.predict(X_val)
                    self.rmse = np.sqrt(mean_squared_error(y_val, self.pred_val, squared=True))
                    self.performance_metrics = {"n_estimators": self.n_estimator,
                                                "criterion": self.criterion,
                                                "oob_score": self.oob_score,
                                                "rmse": self.rmse}
                    print(self.performance_metrics)


                    self.df_dict = pd.DataFrame([self.performance_metrics])
                    self.performance_df = pd.concat([self.performance_df, self.df_dict], ignore_index=True)
        self.performance_df.to_excel(f"./outputs/{response_item}_rfe_hyperparams.xlsx", index=False)
        self.best_params = self.performance_df.sort_values(by="rmse", ascending=True).head(1).to_dict(orient="records")[0]

        self.best_params_fit = self.best_params.copy()
        del self.best_params_fit["rmse"]
        self.best_model = RandomForestRegressor(**self.best_params_fit)
        self.best_model.fit(X_train, y_train)
        
        with open(f"./models/{response_item}_rfe_hypertuned.bin", "wb") as f_out:
             joblib.dump(self.best_model, f_out, compress=3)

        print(f"{response_item} Hyperparameter Tune Random Forest Model available")

        del self.performance_df, self.best_params_fit
        gc.collect()
        return {"model": self.best_model.__class__.__name__, "rmse": self.best_params["rmse"]}


    def extratree_hyper_tuned(self, X_train, y_train, X_val, y_val, response_item:str):
        """Tune the hyper parameters of Extra Tree Regression
        Choose the best params based on Validation Data
        Save the best model based on Validation Data
        
        input: Dict, array, str
        response: dict, binary file"""


        self.n_estimators = range(10, 150, 10)
        self.criterions = ["squared_error", "friedman_mse"]
        self.bootstraps = [True]
        self.oob_scores = [True]
        
        self.performance_df = pd.DataFrame()

        for self.n_estimator in self.n_estimators:
            for self.criterion in self.criterions:
                for self.bootstrap in self.bootstraps:
                    for self.oob_score in self.oob_scores:

                        self.model = ExtraTreesRegressor(n_estimators=self.n_estimator,
                                                        criterion=self.criterion,
                                                        bootstrap=self.bootstrap,
                                                        oob_score=self.oob_score,
                                                        n_jobs=-1,
                                                        verbose=2,
                                                        random_state=143)
                        self.model.fit(X_train, y_train)
                        self.pred_val = self.model.predict(X_val)
                        self.rmse = np.sqrt(mean_squared_error(y_val, self.pred_val, squared=True))
                        self.performance_metrics = {"n_estimators": self.n_estimator,
                                                    "criterion": self.criterion,
                                                    "bootstrap": self.bootstrap,
                                                    "oob_score": self.oob_score,
                                                    "rmse": self.rmse}
                        print(self.performance_metrics)


                        self.df_dict = pd.DataFrame([self.performance_metrics])
                        self.performance_df = pd.concat([self.performance_df, self.df_dict], ignore_index=True)
        self.performance_df.to_excel(f"./outputs/{response_item}_xtratree_hyperparams.xlsx", index=False)
        self.best_params = self.performance_df.sort_values(by="rmse", ascending=True).head(1).to_dict(orient="records")[0]

        self.best_params_fit = self.best_params.copy()
        del self.best_params_fit["rmse"]
        self.best_model = ExtraTreesRegressor(**self.best_params_fit)
        self.best_model.fit(X_train, y_train)
        
        with open(f"./models/{response_item}_xtratree_hypertuned.bin", "wb") as f_out:
             joblib.dump(self.best_model, f_out, compress=3)

        print(f"{response_item} Hyperparameter Tune  Extra Trees Model available")

        del self.performance_df, self.best_params_fit
        gc.collect()
        return {"model": self.best_model.__class__.__name__, "rmse": self.best_params["rmse"]}
    

    def xgb_hyper_tuned(self, X_train, y_train, X_val, y_val, response_item:str):
        """Tune the hyper parameters of XGBoost Regression
        Choose the best params based on Validation Data
        Save the best model based on Validation Data
        
        input: Dict, array, str
        response: dict, binary file"""


        self.boosters = ["gbtree", "dart"]
        self.learning_rates = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75]
        
        self.performance_df = pd.DataFrame()

        for self.booster in  self.boosters:
            for self.lr in self.learning_rates:
                self.model = XGBRegressor(booster=self.booster,
                              eta=self.lr,
                              random_state=143)
                self.model.fit(X_train, y_train)
                self.pred_val = self.model.predict(X_val)
                self.rmse = np.sqrt(mean_squared_error(y_val, self.pred_val, squared=True))
                self.performance_metrics = {"booster": self.booster,
                                            "learning_rate": self.lr,
                                            "rmse": self.rmse}
                print(self.performance_metrics)


                self.df_dict = pd.DataFrame([self.performance_metrics])
                self.performance_df = pd.concat([self.performance_df, self.df_dict], ignore_index=True)
        self.performance_df.to_excel(f"./outputs/{response_item}_xgb_hyperparams.xlsx", index=False)
        self.best_params = self.performance_df.sort_values(by="rmse", ascending=True).head(1).to_dict(orient="records")[0]

        self.best_params_fit = self.best_params.copy()
        del self.best_params_fit["rmse"]
        self.best_model = XGBRegressor(**self.best_params_fit, random_state=143)
        self.best_model.fit(X_train, y_train)
        
        with open(f"./models/{response_item}_xgb_hypertuned.bin", "wb") as f_out:
             joblib.dump(self.best_model, f_out, compress=3)

        print(f"{response_item} Hyperparameter Tune  XGB Model available")

        del self.performance_df, self.best_params_fit
        gc.collect()
        return {"model": self.best_model.__class__.__name__, "rmse": self.best_params["rmse"]}

    def gbr_hyper_tuned(self, X_train, y_train, X_val, y_val, response_item:str):
        """Tune the hyper parameters of Gradient Boosting Regression
        Choose the best params based on Validation Data
        Save the best model based on Validation Data
        
        input: Dict, array, str
        response: dict, binary file"""


        self.losses = ["squared_error", "huber"]
        self.learning_rates = [0.01, 0.1, ]#0.2, 0.3, 0.5, 0.75]
        self.n_estimators = range(10, 20, 10)
        self.criteria = ["squared_error", "friedman_mse"]
        
        self.performance_df = pd.DataFrame()

        for self.loss in  self.losses:
            for self.lr in self.learning_rates:
                for self.criterion in self.criteria:
                    for self.n_estimator in self.n_estimators:
                        self.model = GradientBoostingRegressor(loss=self.loss,
                                                               learning_rate=self.lr,
                                                               criterion=self.criterion,
                                                               n_estimators=self.n_estimator,
                                                               n_iter_no_change=30,
                                                               warm_start=True,
                                                               random_state=143)
                        self.model.fit(X_train, y_train)
                        self.pred_val = self.model.predict(X_val)
                        self.rmse = np.sqrt(mean_squared_error(y_val, self.pred_val, squared=True))
                        self.performance_metrics = {"loss": self.loss,
                                                    "learning_rate": self.lr,
                                                    "criterion": self.criterion,
                                                    "n_estimators": self.n_estimator,
                                                    "n_iter_no_change": 30,
                                                    "warm_start": 30,
                                                    "rmse": self.rmse}
                        print(self.performance_metrics)


                        self.df_dict = pd.DataFrame([self.performance_metrics])
                        self.performance_df = pd.concat([self.performance_df, self.df_dict], ignore_index=True)
        self.performance_df.to_excel(f"./outputs/{response_item}_gbr_hyperparams.xlsx", index=False)
        self.best_params = self.performance_df.sort_values(by="rmse", ascending=True).head(1).to_dict(orient="records")[0]

        self.best_params_fit = self.best_params.copy()
        del self.best_params_fit["rmse"]
        self.best_model = GradientBoostingRegressor(**self.best_params_fit, random_state=143)
        self.best_model.fit(X_train, y_train)
        
        with open(f"./models/{response_item}_gbr_hypertuned.bin", "wb") as f_out:
             joblib.dump(self.best_model, f_out, compress=3)

        print(f"{response_item} Hyperparameter Tune  GBR Model available")

        del self.performance_df, self.best_params_fit
        gc.collect()
        return {"model": self.best_model.__class__.__name__, "rmse": self.best_params["rmse"]}




if __name__ == "__main__":
    #   train data
    print("#######################  TRAIN DATA PREP     ########################")
    print("\n")
    df = pd.read_parquet("./datasets/yellow_tripdata_2022-01.parquet")
    prep = Preprocessor(df)
    prep.lower_colnames()
    print("lower names done")
    df = prep.feature_cleanup()
    print("feature cleanup done")
    prep.ohe_fit(df)
    print("ohe fit done")
    df = prep.ohe_transform()
    print("ohe transform done done")
    df = prep.impute_missing_values(df)
    print("NAN impute done")
    X_train, y_train_fare, y_train_duration = prep.create_predictor_response()
    print("Pred response done")
    prep.vectorizer_fit()
    print("Dv fit done")    
    X_train = prep.vectorizer_transform()
    print("DV transform done")
    print(X_train)
    print(y_train_fare)

    #   validation data
    print("#######################  VALIDATION DATA PREP    ########################")
    print("\n")
    df = pd.read_parquet("./datasets/yellow_tripdata_2022-02.parquet")
    prep = Preprocessor(df)
    prep.lower_colnames()
    print("lower names done")
    df = prep.feature_cleanup()
    print("feature cleanup done")
    print("ohe fit done")
    df = prep.ohe_transform()
    print("ohe transform done done")
    df = prep.impute_missing_values(df)
    print("NAN impute done")
    X_val, y_val_fare, y_val_duration = prep.create_predictor_response()
    print("Pred response done")
    print("Dv fit done")    
    X_val = prep.vectorizer_transform()
    print("DV transform done")
    print(X_val)
    print(y_val_fare)

    print("#######################  FARE HYPERPARAMETER TUNING - RIDGE  ########################")
    print("\n")
    
    print("Fare Prediction Ridge")
    h = HyperOpt()
    fare_ridge_metrics = h.ridge_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_ridge_metrics)

    print("#######################  DURATION HYPERPARAMETER TUNING - RIDGE  ########################")
    print("\n")
    print("Duration Prediction Validation")
    dur_ridge_metrics = h.ridge_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_ridge_metrics)

    print("#######################  FARE HYPERPARAMETER TUNING - RANDOM FOREST  ########################")
    print("\n")
    
    print("Fare Prediction Validation RFE")
    h = HyperOpt()
    fare_rfe_metrics = h.randomforest_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_rfe_metrics)

    print("#######################  DURATION HYPERPARAMETER TUNING - RANDOM FOREST  ########################")
    print("\n")
    print("Duration Prediction Validation RFE")
    dur_rfe_metrics = h.randomforest_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_rfe_metrics)

    print("#######################  FARE HYPERPARAMETER TUNING - XTRA TREE  ########################")
    print("\n")
    
    print("Fare Prediction Validation XTE")
    h = HyperOpt()
    fare_xtra_metrics = h.extratree_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_xtra_metrics)

    print("#######################  DURATION HYPERPARAMETER TUNING - XTRA TREE  ########################")
    print("\n")
    print("Duration Prediction Validation XTE")
    dur_xtra_metrics = h.extratree_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_xtra_metrics)


    print("#######################  FARE HYPERPARAMETER TUNING - XGB  ########################")
    print("\n")
    
    print("Fare Prediction Validation XGB")
    h = HyperOpt()
    fare_xgb_metrics = h.xgb_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_xgb_metrics)

    print("#######################  DURATION HYPERPARAMETER TUNING - XGB  ########################")
    print("\n")
    print("Duration Prediction Validation XGB")
    dur_xgb_metrics = h.xgb_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_xgb_metrics)


    print("#######################  FARE HYPERPARAMETER TUNING - GBR  ########################")
    print("\n")
    
    print("Fare Prediction Validation GBR")
    h = HyperOpt()
    fare_gbr_metrics = h.gbr_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_gbr_metrics)

    print("#######################  DURATION HYPERPARAMETER TUNING - GBR  ########################")
    print("\n")
    print("Duration Prediction Validation GBR")
    dur_gbr_metrics = h.gbr_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_gbr_metrics)



##############################################################
    with open("./outputs/best_fare_ridge.json", "w") as f_out:
        json.dump(fare_ridge_metrics, f_out)
    with open("./outputs/best_fare_rfe.json", "w") as f_out:
        json.dump(fare_rfe_metrics, f_out)
    with open("./outputs/best_fare_xtra.json", "w") as f_out:
        json.dump(fare_xtra_metrics, f_out)
    with open("./outputs/best_fare_xgb.json", "w") as f_out:
        json.dump(fare_xgb_metrics, f_out)
    with open("./outputs/best_fare_gbr.json", "w") as f_out:
        json.dump(fare_gbr_metrics, f_out)

    with open("./outputs/best_dur_ridge.json", "w") as f_out:
        json.dump(dur_ridge_metrics, f_out)
    with open("./outputs/best_dur_rfe.json", "w") as f_out:
        json.dump(dur_rfe_metrics, f_out)
    with open("./outputs/best_dur_xtra.json", "w") as f_out:
        json.dump(dur_xtra_metrics, f_out)
    with open("./outputs/best_dur_xgb.json", "w") as f_out:
        json.dump(dur_xgb_metrics, f_out)
    with open("./outputs/best_dur_gbr.json", "w") as f_out:
        json.dump(dur_gbr_metrics, f_out)
