#!/usr/bin/env python3
import json
import numpy as np
from azure.storage.blob import BlobServiceClient

BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=<storage>;AccountKey=<KEY>;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "results"
BLOB_NAME = "C_matrix.json"

def upload_matrix(C: np.ndarray):
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)
    payload = json.dumps(C.tolist())
    blob_client.upload_blob(payload, overwrite=True)
    print(f"Uploadovan blob: {CONTAINER_NAME}/{BLOB_NAME}")

if __name__ == "__main__":
    C = np.random.rand(4, 4).astype(np.float32)
    upload_matrix(C)