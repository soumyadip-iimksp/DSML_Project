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

## Tableau Visualization of Metrics and Hyperparameters 
<br>
**Fare Prediction Model Metrics Visualization** 
[link](https://public.tableau.com/views/DS_ML_Project_tableau_Dashboards_Fare_Prediction/FarePredictionDashboard?:language=en-US&:display_count=n&:origin=viz_share_link)
<img width="1258" alt="Screenshot 2023-03-03 150646" src="https://user-images.githubusercontent.com/121397382/222685889-8caf3148-11ff-4300-a1f6-eebf8ebf142f.png">
<br>
<br>
**Duration Prediction Model Metrics Visualization** 
[link](https://public.tableau.com/shared/DHSYWHQGP?:display_count=n&:origin=viz_share_link)
<img width="1260" alt="Screenshot 2023-03-03 150808" src="https://user-images.githubusercontent.com/121397382/222685898-704d377c-f3c6-41c4-a3f5-238ed1eefdc3.png">


