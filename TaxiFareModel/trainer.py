# imports
from re import M
from typing_extensions import Self
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property

import mlflow
from  mlflow.tracking import MlflowClient

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso


Models = {
    "LinearRegression": LinearRegression(),
    "RandomForrest": RandomForestRegressor(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SGDRegressor": SGDRegressor()
}

class Trainer():

    def __init__(self, X, y, model= "RandomForest"):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[DE] [Berlin] [jahlah] TaxiFare v1"
        self.model_name = model
        self.model = RandomForestRegressor()
        #self.model = Models[model]


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (self.model_name, self.model)
        ])

        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        #print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis = 1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Run for just one model to test
    # train
    #trainer = Trainer(X_train, y_train, "LinearRegression")
    #trainer.run()
    # evaluate
    #rmse = trainer.evaluate(X_test, y_test)
    #trainer.mlflow_log_param("model", trainer.model_name)
    #trainer.mlflow_log_metric("rmse", rmse)

    # Run multiple models to find best model tpye
    #for model in Models:
    #    trainer = Trainer(X_train, y_train, model)
    #    trainer.run()
    #    rmse = trainer.evaluate(X_test, y_test)

    #    trainer.mlflow_log_param("model", trainer.model_name)
    #    trainer.mlflow_log_metric("rmse", rmse)

    # Cross validation for model selection
    for model in Models:
        trainer = Trainer(X_train, y_train, model)
        trainer.set_pipeline()
        cv = cross_validate(trainer.pipeline,
                            X_train,
                            y_train,
                            cv = 5,
                            n_jobs = -1,
                            scoring = ("r2", "neg_mean_squared_error"))

        rmse = float(cv["test_neg_mean_squared_error"].mean())
        trainer.mlflow_log_param("model", trainer.model_name)
        trainer.mlflow_log_metric("CV rmse", rmse)
