__author__ = "Soumyadip Majumder"
__version__ = "1.0.1"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "21 Feb 2023"

import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor


class Preprocessor:

    def __init__(self, df) -> None:
        self.df = df
        

    def lower_colnames(self):
        """ To standardize the col names to lower
        input: dataframe
        response: dataframe
        """
        self.col_names = list(self.df.columns)
        self.col_names = [self.col.lower() for self.col in self.col_names]
        self.df.columns = self.col_names
        
        return self.df
    

    def feature_cleanup(self):
        """ To create new feature from existing features 
        and remove unwanted columns
        
        duration: tpep_dropoff_datetime - tpep_pickup_datetime
        
        drop:
        vendorid
        tpep_dropoff_datetime
        tpep_pickup_datetime

        input: dataframe
        response dataframe
        """
        self.df = self.lower_colnames()

        self.df["tpep_pickup_datetime"] = pd.to_datetime(self.df["tpep_pickup_datetime"])
        self.df["tpep_dropoff_datetime"] = pd.to_datetime(self.df["tpep_dropoff_datetime"])
        self.df["duration"] = duration = self.df["tpep_dropoff_datetime"] - self.df["tpep_pickup_datetime"]
        self.df["duration"] = self.df["duration"].apply(lambda td: td.total_seconds()/60)

        self.df.drop(columns = ["vendorid", "tpep_pickup_datetime", "tpep_dropoff_datetime"], axis=1, inplace=True)
        
        return self.df


    def ohe_fit(self, data):
        """Save OHE model as pickle for transforming validation and test datasets
        input: dataframe
        response: pickle file"""
        self.data = data
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(self.data["store_and_fwd_flag"].values.reshape(-1,1))
        print(self.ohe.categories_)
        with open("./models/ohe.bin", "wb") as f_out:
            pkl.dump(self.ohe, f_out)


    def ohe_transform(self):
        """One hot encode the categorical variable
        input: dataframe
        response: dataframe
        """

        with open("./models/ohe.bin", "rb") as f_in:
            self.ohe_model = pkl.load(f_in)

            self.sop = self.ohe_model.transform(self.df["store_and_fwd_flag"].values.reshape(-1,1)).toarray()
            self.df = pd.concat([self.df, pd.DataFrame(self.sop, columns=["store_and_fwd_flag_no", 
                                                "store_and_fwd_flag_yes", 
                                                "store_and_fwd_flag_none"])], axis=1)
            
            self.df.drop("store_and_fwd_flag", axis=1, inplace=True)
            #self.df.to_csv("test_op.csv")
            return self.df
        

    def create_predictor_response(self):
        """splits the data into predictores and response
        input: dataframe
        response:
                dataframe
                array
                array
        """

        self.y_fare = self.df["total_amount"].values
        self.y_duration = self.df["duration"].values

        self.X = self.df.drop(columns=["total_amount","duration"], axis=1)

        return self.X, self.y_fare, self.y_duration
    

    def impute_missing_values(self, data:pd.DataFrame):
        """Impute the missing values in the dataframe withh KNN
        number of neighbours = square root of of total number of observations
        input: dataframe
        response: dataframe
        """
        self.data = data
        nan_cols = [i for i in self.data.columns if self.data[i].isnull().any()]

        for col in nan_cols:
            #nan_imputer = KNNImputer(n_neighbors=int(np.sqrt(len(data))), weights="uniform")
            nan_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            self.data[col] = nan_imputer.fit_transform(self.data[col].values.reshape(-1,1))
            print(f"{col} colum imputed")
        return self.data
        

    def vectorizer_fit(self):
        """"""
        self.X_for_fit = self.X.to_dict(orient="records")

        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(self.X_for_fit)

        with open("./models/dict_vec.bin", "wb") as f_out:
            pkl.dump(self.dv, f_out)


    def vectorizer_transform(self):
        """Apply Dictionary vectorizer transfromation to 
        train, validation and test dataset
        """
        with open("./models/dict_vec.bin", "rb") as f_in:
            self.dv_model = pkl.load(f_in)
        
        self.X, _, _ = self.create_predictor_response()

        self.X = self.X.to_dict(orient="records")

        self.X = self.dv_model.transform(self.X)
        return self.X

            

if __name__ == "__main__":

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
    X, y_fare, y_duration = prep.create_predictor_response()
    print("Pred response done")
    prep.vectorizer_fit()
    print("Dv fit done")    
    X = prep.vectorizer_transform()
    print("DV transform done")
    print(X)
    print(y_fare)
    rfe = RandomForestRegressor(verbose=1, n_jobs=-1)
    rfe.fit(X, y_fare)
    print("RFE fit done")
    print(rfe.predict(X))