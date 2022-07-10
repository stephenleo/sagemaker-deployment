
import argparse
import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

model_file_name = "pipeline_model.joblib"

# Main function
def main():
    logger.info("Starting training")
    parser = argparse.ArgumentParser()
    
    # Inbuilt Arguments: https://github.com/aws/sagemaker-containers#id11
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    
    # Custom Arguments
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=float, default=0.05)
    parser.add_argument("--criterion", type=str, default="gini")
    
    
    args, _ = parser.parse_known_args()
    
    logger.info("Load data")
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.validation, "validation.csv"))

    # Define the columns
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    cont_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Split X(features) and y(response)
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # One hot encode the categorical columns
    ohe = OneHotEncoder(drop="first")

    # Scale the continuous columns
    sc = StandardScaler()

    # Column transformer to apply transformations on both categorical and continuous columns
    ct = ColumnTransformer([
        ("One Hot Encoding", ohe, cat_cols),
        ("Scaling", sc, cont_cols)
    ])
    
    logger.info("Train the model pipeline")
    rfc = RandomForestClassifier(n_estimators=args.n_estimators, 
                                 min_samples_split=args.min_samples_split, 
                                 criterion=args.criterion)

    # Sklearn pipeline
    pipeline_rfc_model = Pipeline([
        ("Data Transformations", ct),
        ("Random Forest Model", rfc)
    ])

    # Fit the model locally on a smaller subset of data
    pipeline_rfc_model.fit(X_train, y_train)

    # Check the accuracy on training data
    train_accuracy = pipeline_rfc_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Check the accuracy on test data
    test_accuracy = pipeline_rfc_model.score(X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    logger.info("Save the model")
    model_save_path = os.path.join(args.model_dir, model_file_name)
    joblib.dump(pipeline_rfc_model, model_save_path)
    print(f"Model saved at {model_save_path}")

# Run the main function when the script runs
if __name__ == "__main__":
    main()
