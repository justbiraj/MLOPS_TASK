import mlflow
import mlflow.sklearn
from mlops.train_model import train_model
from mlops.evaluate import evaluate_model
from mlops.utils import init_mlflow

def main():
    init_mlflow()
    mlflow.set_experiment("mini-mlops") 

    n_estimators = 100
    max_depth = 3
    random_state = 42

    with mlflow.start_run():
        model, X_test, y_test = train_model(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

        acc = evaluate_model(model, X_test, y_test)

        # Log params, metrics, and model
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.sklearn.log_model(model, "model")

        print(f" Model trained and logged successfully | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
