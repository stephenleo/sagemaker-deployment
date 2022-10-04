"""Evaluation script"""
import json
import logging
import pathlib
import joblib
import tarfile
import pandas as pd

from sklearn.metrics import confusion_matrix, get_scorer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")

    logger.info("Loading model.")
    model_file_name = "pipeline_model.joblib"
    pipeline_model = joblib.load(model_file_name)

    logger.info("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]   
    
    logger.info("Writing out metrics")
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # How to specify model metrics
    # https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            metric: {
                "value": get_scorer(metric)(pipeline_model, X_test, y_test),
                "standard_deviation": "NaN"
            } for metric in ["accuracy", "precision", "recall", "f1"]
        }
    }
    
    y_pred = pipeline_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    report_dict["binary_classification_metrics"]["confusion_matrix"] = {
        str(row): {
            str(col): cm[row][col]
            for col in range(len(cm[row]))
        }
        for row in range(len(cm))
    }

    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
