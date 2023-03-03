# DSML_Project
NY taxi Fare Prediction and Model Optimization

### To *RUN* the code:
`pip install -r requirements.txt`

### To Train the model and for Hyperparameter Tuning
`python train.py`

### To test the model
`python test.py`

**Note:** Below files are not available due to large size. Can be created by running the `train.py` script
```
models/fare_xtratree_hypertuned.bin
models/fare_rfe_hypertuned.bin
models/duration_xtratree_hypertuned.bin
models/duration_rfe_hypertuned.bin
```

**Training Data:** [datasets\yellow_tripdata_2022-01.parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet) <br>
**Validation Data:** [datasets\yellow_tripdata_2022-02.parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet) <br>
**Test Data:** [datasets\yellow_tripdata_2022-03.parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-03.parquet) <br>