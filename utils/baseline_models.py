__author__ = "Soumyadip Majumder"
__version__ = "1.0.0"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "21 Feb 2023"

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from utils.preprocessing import Preprocessor

class Models:

    def __init__(self) -> None:
        pass

    def train_validate_fare(self, X_train, y_train, X_val, y_val, response_item:str):
        """Train different models 
        input: dictionary and list
        response: binary files (model weights)
        """

        self.model_classes = [LinearRegression(n_jobs=-1), 
                            Lasso(random_state=143),
                            Ridge(random_state=143),
                            ElasticNet(l1_ratio=0.5, random_state=143) ,
                            RandomForestRegressor(n_jobs=-1, verbose=1, random_state=143), 
                            GradientBoostingRegressor(learning_rate=0.01, verbose=1, random_state=143),
                            ExtraTreesRegressor(n_jobs=-1, random_state=143, verbose=1), 
                            XGBRegressor()]
        
        self.model_lst = []
        self.rmse_lst = []

        for self.model_class in self.model_classes:
        
            self.model_name = self.model_class.__class__.__name__
            print(self.model_name)
        
            self.model_class.fit(X_train, y_train)
        
            self.pred_val = self.model_class.predict(X_val)
            self.rmse = np.sqrt(mean_squared_error(y_val, self.pred_val, squared=True))
            print("RMSE:", self.rmse)
        
            self.model_lst.append(self.model_name)
            self.rmse_lst.append(self.rmse)
        
        self.df_baseline_models =  pd.DataFrame({"model": self.model_lst, "rmse": self.rmse_lst}) 
        self.df_baseline_models.to_csv(f"{response_item}_models_baseline_rmse.csv", index=False)
        
        return self.df_baseline_models


if __name__ == "__main__":

    #   train data
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
    print("Validation")
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

    print("Fare Prediction Validation")
    m = Models()
    fare_df_bl = m.train_validate_fare(X_train, y_train_fare, X_val, y_val_fare, "fare")

    print("Duration Prediction Validation")
    m = Models()
    duration_df_bl = m.train_validate_fare(X_train, y_train_duration, X_val, y_val_duration, "duration")

    #   Chose the top four models
    fare_df_bl_t4 = fare_df_bl.sort_values(by=["rmse"], ascending=True).head(4)
    duration_df_bl_t4 = duration_df_bl.sort_values(by=["rmse"], ascending=True).head(4)
