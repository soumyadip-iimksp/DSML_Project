__author__ = "Soumyadip Majumder"
__version__ = "1.0.2"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "02 Mar 2023"

import pandas as pd
import joblib
import json
from pathlib import Path
from utils.preprocessing import Preprocessor


def select_best_model(model_item:str):
    
    if model_item == "dur": 
        paths = Path("./outputs/").glob(f"*{model_item}*.json")
        model_type = "duration"
        model_df = pd.DataFrame([pd.read_json(p, typ="series") for p in paths])
        best_model = model_df.sort_values(by="rmse", ascending=True).head(1)["model"]
    elif model_item == "fare":
        paths = Path("./outputs/").glob(f"*{model_item}*.json")
        model_type = model_item
        model_df = pd.DataFrame([pd.read_json(p, typ="series") for p in paths])
        best_model = model_df.sort_values(by="rmse", ascending=True).head(1)["model"]
    else:
        print("Incorrect details")
    
    print(best_model)
    if str(best_model) == "XGBRegressor":
        model = joblib.load(f"./models/{model_type}_xgb_hypertuned.bin")
    elif str(best_model) == "ExtraTreesRegressor":
        model = joblib.load(f"./models/{model_type}_xtra_hypertuned.bin")
    elif str(best_model) == "GradientBoostingRegressor":
        model = joblib.load(f"./models/{model_type}_gbr_hypertuned.bin")
    elif str(best_model) == "Ridge":
        model = joblib.load(f"./models/{model_type}_ridge_hypertuned.bin")
    else:
        model = joblib.load(f"./models/{model_type}_rfe_hypertuned.bin") 

    return model

def pred_output(X, model_item:str):

    model = select_best_model(model_item=model_item)
    op = model.predict(X)
    print(op)
    with open(f"./predictions/{model_item}_test_prediction.json", 'w') as f_out:
        json.dump(op.tolist(), f_out)

    return "Prediction Output Generated"




if __name__ == "__main__":
    
    df = pd.read_parquet("./datasets/yellow_tripdata_2022-03.parquet")
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
    X, y_fare, y_duration = prep.create_predictor_response()
    print("Pred response done")
    prep.vectorizer_fit()
    print("Dv fit done")    
    X = prep.vectorizer_transform()

    # Generate Prediction
    pred_output(X, "fare")
    print("********* Fare Predicted *************")
    pred_output(X, "dur")
    print("********* Duration Predicted *************")