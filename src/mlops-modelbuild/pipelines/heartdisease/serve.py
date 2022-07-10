
import os
import joblib
import pandas as pd

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Load the Model
def model_fn(model_dir):
    model_file_name = "pipeline_model.joblib"
    pipeline_model = joblib.load(os.path.join(model_dir, model_file_name))
    
    return pipeline_model

# Load the input data
def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    if request_content_type == "application/json":
        input_object = pd.read_json(request_body, lines=True)
        
        return input_object
    else:
        raise ValueError("Only application/json content type supported!")

def predict_fn(input_object, pipeline_model):
    predictions = pipeline_model.predict(input_object)
    pred_probs = pipeline_model.predict_proba(input_object)
    
    prediction_object = pd.DataFrame(
        {
            "prediction": predictions.tolist(),
            "pred_prob_class0": pred_probs[:, 0].tolist(),
            "pred_prob_class1": pred_probs[:, 1].tolist()
        }
    )
    
    return prediction_object

def output_fn(prediction_object, request_content_type):
    return_object = prediction_object.to_json(orient="records", lines=True)
    
    return return_object
