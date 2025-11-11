from fastapi import FastAPI
from api.schemas import IrisInput
from sklearn.datasets import load_iris
import numpy as np
import mlflow
import mlflow.sklearn
import os
from mlflow.tracking import MlflowClient


app = FastAPI(title="Mini MLOps Model API")
client = MlflowClient()

def get_latest_run_id(experiment_name ="mini-mlops"):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"])
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")
    return runs[0].info.run_id




# print(get_latest_run_id(experiment_name="mini-mlops"))
# === Initialize Model ===
try:
    latest_run_id = get_latest_run_id()
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from MLflow Run ID: {latest_run_id}")
except Exception as e:
    print(f"Could not load model: {e}")
    model = None


@app.post("/predict")
def predict(input: IrisInput):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    data = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    prediction = model.predict(data)[0]

    # Default species mapping: use sklearn's iris target names (trained with load_iris)
    species = None
    try:
        iris = load_iris()
        # prediction may be numpy types; cast to int for indexing
        species = iris.target_names[int(prediction)]
    except Exception:
        # Fallbacks: if the model stores string class labels in classes_, use that
        try:
            classes = getattr(model, "classes_", None)
            if classes is not None:
                # If classes_ contains string labels
                if np.issubdtype(classes.dtype, np.str_):
                    species = str(classes[int(prediction)])
        except Exception:
            species = None

    result = {"prediction": int(prediction)}
    if species is not None:
        result["species"] = species
    return result
