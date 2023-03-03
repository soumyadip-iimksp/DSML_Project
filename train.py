__author__ = "Soumyadip Majumder"
__version__ = "1.0.4"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "28 Feb 2023"

import pandas as pd
import json
import gc

from utils.preprocessing import Preprocessor
from utils.baseline_models import Models
from utils.tune_hyperparameters import HyperOpt

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

    print("#######################  FARE PREDICTION     ########################")
    print("\n")
    print("Fare Prediction Validation")
    m = Models()
    m.train_validate_fare(X_train, y_train_fare, X_val, y_val_fare, "fare")

    del df
    gc.collect()

    print("#######################  DURATION PREDICTION     ########################")
    print("\n")
    print("Duration Prediction Validation")
    m = Models()
    m.train_validate_fare(X_train, y_train_duration, X_val, y_val_duration, "duration")

    
    print("#######################  FARE HYPERPARAMETER TUNING - RIDGE  ########################")
    print("\n")
    
    print("Fare Prediction Ridge")
    h = HyperOpt()
    fare_ridge_metrics = h.ridge_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_ridge_metrics)

    with open("./outputs/best_fare_ridge.json", "w") as f_out:
        json.dump(fare_ridge_metrics, f_out)
    del fare_ridge_metrics
    gc.collect()

    print("#######################  DURATION HYPERPARAMETER TUNING - RIDGE  ########################")
    print("\n")
    print("Duration Prediction Validation")
    dur_ridge_metrics = h.ridge_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_ridge_metrics)

    with open("./outputs/best_dur_ridge.json", "w") as f_out:
        json.dump(dur_ridge_metrics, f_out)
    del dur_ridge_metrics
    gc.collect()

    print("#######################  FARE HYPERPARAMETER TUNING - RANDOM FOREST  ########################")
    print("\n")
    
    print("Fare Prediction Validation RFE")
    h = HyperOpt()
    fare_rfe_metrics = h.randomforest_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_rfe_metrics)
    
    with open("./outputs/best_fare_rfe.json", "w") as f_out:
        json.dump(fare_rfe_metrics, f_out)
    del fare_rfe_metrics
    gc.collect()

    print("#######################  DURATION HYPERPARAMETER TUNING - RANDOM FOREST  ########################")
    print("\n")
    print("Duration Prediction Validation RFE")
    dur_rfe_metrics = h.randomforest_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_rfe_metrics)

    with open("./outputs/best_dur_rfe.json", "w") as f_out:
        json.dump(dur_rfe_metrics, f_out)
    del dur_rfe_metrics
    gc.collect()

    print("#######################  FARE HYPERPARAMETER TUNING - XTRA TREE  ########################")
    print("\n")
    
    print("Fare Prediction Validation XTE")
    h = HyperOpt()
    fare_xtra_metrics = h.extratree_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_xtra_metrics)

    with open("./outputs/best_fare_xtra.json", "w") as f_out:
        json.dump(fare_xtra_metrics, f_out)
    del fare_xtra_metrics
    gc.collect()

    print("#######################  DURATION HYPERPARAMETER TUNING - XTRA TREE  ########################")
    print("\n")
    print("Duration Prediction Validation XTE")
    dur_xtra_metrics = h.extratree_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_xtra_metrics)

    with open("./outputs/best_dur_xtra.json", "w") as f_out:
        json.dump(dur_xtra_metrics, f_out)
    del dur_xtra_metrics
    gc.collect()

    print("#######################  FARE HYPERPARAMETER TUNING - XGB  ########################")
    print("\n")
    
    print("Fare Prediction Validation XGB")
    h = HyperOpt()
    fare_xgb_metrics = h.xgb_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_xgb_metrics)

    with open("./outputs/best_fare_xgb.json", "w") as f_out:
        json.dump(fare_xgb_metrics, f_out)
    del fare_xgb_metrics
    gc.collect()

    print("#######################  DURATION HYPERPARAMETER TUNING - XGB  ########################")
    print("\n")
    print("Duration Prediction Validation XGB")
    dur_xgb_metrics = h.xgb_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_xgb_metrics)

    with open("./outputs/best_dur_xgb.json", "w") as f_out:
        json.dump(dur_xgb_metrics, f_out)
    del dur_xgb_metrics
    gc.collect()

    print("#######################  FARE HYPERPARAMETER TUNING - GBR  ########################")
    print("\n")
    
    print("Fare Prediction Validation GBR")
    h = HyperOpt()
    fare_gbr_metrics = h.gbr_hyper_tuned(X_train, y_train_fare, X_val, y_val_fare, "fare")
    print(fare_gbr_metrics)

    with open("./outputs/best_fare_gbr.json", "w") as f_out:
        json.dump(fare_gbr_metrics, f_out)
    del fare_gbr_metrics
    gc.collect()

    print("#######################  DURATION HYPERPARAMETER TUNING - GBR  ########################")
    print("\n")
    print("Duration Prediction Validation GBR")
    dur_gbr_metrics = h.gbr_hyper_tuned(X_train, y_train_duration, X_val, y_val_duration, "duration")
    print(dur_gbr_metrics)
    with open("./outputs/best_dur_gbr.json", "w") as f_out:
        json.dump(dur_gbr_metrics, f_out)
    del dur_gbr_metrics
    gc.collect()