import mlflow
import mlflow.sklearn

def init_mlflow():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mini-mlops")
