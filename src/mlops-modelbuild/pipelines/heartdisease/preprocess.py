"""Create the training and testing dataset."""
import argparse
import logging
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main():
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="URL of input data")
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    logger.info("Reading data")
    heart_df = pd.read_csv(args.input_data, header=None)
    heart_df.columns = ["age", "sex", "cp", "trestbps", "chol", 
                        "fbs", "restecg", "thalach", "exang",
                        "oldpeak", "slope", "ca", "thal", "target"]
    
    logger.info("Basic cleanup")
    heart_df["target"] = heart_df["target"].apply(lambda x: 1 if x>0 else 0)
    heart_df = heart_df[(heart_df["ca"]!="?") & (heart_df["thal"]!="?")]

    logger.info("Split out training and testing datasets")
    train_df, test_df = train_test_split(heart_df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    logger.info(f"Writing out datasets to {base_dir}")
    train_df.to_csv(f"{base_dir}/train/train.csv", index=False)
    val_df.to_csv(f"{base_dir}/validation/validation.csv", index=False)
    test_df.to_csv(f"{base_dir}/test/test.csv", index=False)

if __name__ == "__main__":
    main()