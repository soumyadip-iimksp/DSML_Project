__author__ = "Soumyadip Majumder"
__version__ = "1.0.1"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "21 Feb 2023"

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from preprocessing import Preprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class Models:

    def __init__(self) -> None:
        pass

    def train(self, X, y):
        """Train different models 
        input: dictionary and list
        response: binary files (model weights)
        """

        model_classes = [LinearRegression(n_jobs=-1), 
                         Lasso(random_state=143),
                         Ridge(random_state=143),
                         ElasticNet(l1_ratio=0.5, random_state=143) ,
                         RandomForestRegressor(n_jobs=-1, verbose=1, random_state=143), 
                         GradientBoostingRegressor(learning_rate=0.01, verbose=1, random_state=143),
                         ExtraTreesRegressor(n_jobs=-1, random_state=143, verbose=1), 
                         XGBRegressor()]
        model_lst = []
        rmse_lst = []
        for model_class in model_classes:
            model_name = model_class.__class__.__name__
            print(model_name)
            model_class.fit(X, y)
            rmse = np.sqrt(mean_squared_error(y, model_class.predict(X), squared=True))
            print("RMSE:", rmse)
            model_lst.append(model_name)
            rmse_lst.append(rmse)
        pd.DataFrame({"model":model_lst, "rmse":rmse_lst}).to_csv(f"models_baseline_rmse.csv", index=False)

if __name__ == "__main__":
    df = pd.read_parquet("./datasets/yellow_tripdata_2022-01.parquet")
    prep = Preprocessor(df)
    prep.lower_colnames()
    print("lower names done")
    df = prep.feature_cleanup()
    print("feature cleanup done")
    #prep.ohe_fit(df)
    print("ohe fit done")
    df = prep.ohe_transform()
    print("ohe transform done done")
    df = prep.impute_missing_values(df)
    print("NAN impute done")
    X, y_fare, y_duration = prep.create_predictor_response()
    print("Pred response done")
    #prep.vectorizer_fit()
    print("Dv fit done")    
    X = prep.vectorizer_transform()
    print("DV transform done")
    print(X)
    print(y_fare)
    m = Models()
    m.train(X, y_fare)