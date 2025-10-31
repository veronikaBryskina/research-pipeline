import fsspec
import os
import csv
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

class DataProcessor:
    def __init__(self) -> None:
        self.fs = fsspec.filesystem(
            "s3",
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            client_kwargs={"endpoint_url": os.getenv("ENDPOINT_URL")},
            config_kwargs={"s3": {"addressing_style": "path"}} 
        )
        self.mlflow_logging_url = f"{os.getenv('ENDPOINT_URL')}/mlflow/"

    def fetch_data(self, file_name: str) -> pd.DataFrame:
        file_path = f"datafiles/{file_name}"
        with self.fs.open(file_path, "rb") as f:
            if str(file_path).endswith(".csv"):
                print("It's a csv!")
                return pd.read_csv(f)
            elif str(file_path).endswith(".json"):
                print("It's a json!")
                return pd.read_json(f)
            elif str(file_path).endswith(".parquet"):
                print("It's a parquet!")
                return pd.read_parquet(f)
            else:
                raise ValueError(f"Unsupported file: {file_path}")
    
    def upload_data(self, file_path: str) -> None:
        self.fs.put(file_path, "datafiles/")
        print("Data successfully loaded")

    def delete_data(self, file_name: str) -> None:
        self.fs.rm(f"datafiles/{file_name}")
        print("Data successfully deleted")

    
    def db_encoding(self, encoder):
        return