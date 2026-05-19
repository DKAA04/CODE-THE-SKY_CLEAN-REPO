"""Upload features DataFrame to BigQuery for BQML training."""
from __future__ import annotations
import pandas as pd
from google.cloud import bigquery
from src.config import GCP_PROJECT_ID, BQ_DATASET, BQ_FEATURES_TABLE


def ensure_dataset(client: bigquery.Client, dataset_id: str = BQ_DATASET):
    ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{dataset_id}")
    ref.location = "EU"
    try:
        client.get_dataset(ref)
    except Exception:
        client.create_dataset(ref)
        print(f"Created dataset {dataset_id}")


def upload_features(features_df: pd.DataFrame, table_id: str = BQ_FEATURES_TABLE):
    client = bigquery.Client(project=GCP_PROJECT_ID)
    ensure_dataset(client)
    table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{table_id}"

    df = features_df.copy()
    df.columns = [c.replace("-", "_").replace(" ", "_").lower() for c in df.columns]

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"Loaded {len(df)} rows to {table_ref}")
    return table_ref
