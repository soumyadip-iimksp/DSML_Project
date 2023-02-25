import pandas as pd
import numpy as np

df = pd.read_parquet("./datasets/yellow_tripdata_2022-01.parquet")

print(df.head())

df.to_csv("test_op.csv", index=False)