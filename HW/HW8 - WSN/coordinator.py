#!/usr/bin/env python3
import asyncio
import numpy as np
import time
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import CloudToDeviceMethod

IOTHUB_CONNECTION_STRING = "HostName=<your-hub>.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=<KEY>"
WORKER_DEVICE_IDS = ["wsn-node-01", "wsn-node-02", "wsn-node-03", "wsn-node-04"]

def partition_matrix(A: np.ndarray, n_workers: int):
    m = A.shape[0]
    chunk = m // n_workers
    parts = []
    for i in range(n_workers):
        row_start = i * chunk
        row_end = row_start + chunk if i < n_workers - 1 else m
        parts.append((row_start, row_end, A[row_start:row_end].tolist()))
    return parts

def invoke_multiply_method(registry_manager, device_id: str, payload: dict):
    method = CloudToDeviceMethod(
        method_name="multiply",
        payload=payload,
        response_timeout_in_seconds=120,
        connect_timeout_in_seconds=30
    )
    response = registry_manager.invoke_device_method(device_id, method)
    return response.payload

async def distributed_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape[1] == B.shape[0], "Nekompatibilne dimenzije matrica"

    n_workers = len(WORKER_DEVICE_IDS)
    m, n = A.shape[0], B.shape[1]
    B_list = B.tolist()
    parts = partition_matrix(A, n_workers)

    registry_manager = IoTHubRegistryManager.from_connection_string(IOTHUB_CONNECTION_STRING)
    loop = asyncio.get_event_loop()

    t0 = time.time()
    futures = []

    for task_id, (device_id, (row_start, row_end, A_chunk)) in enumerate(zip(WORKER_DEVICE_IDS, parts)):
        payload = {
            "task_id": task_id,
            "row_start": row_start,
            "row_end": row_end,
            "A_chunk": A_chunk,
            "B": B_list
        }
        futures.append(
            loop.run_in_executor(None, invoke_multiply_method, registry_manager, device_id, payload)
        )

    responses = await asyncio.gather(*futures)
    elapsed = time.time() - t0
    print(f"[Coordinator] Svi odgovori stigli za {elapsed:.3f}s")

    C = np.zeros((m, n), dtype=np.float32)
    for resp in responses:
        rs = resp["row_start"]
        re = resp["row_end"]
        C_chunk = np.array(resp["C_chunk"], dtype=np.float32)
        C[rs:re, :] = C_chunk

    return C

async def main():
    m, k, n = 120, 100, 80
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)

    C_dist = await distributed_matmul(A, B)
    C_ref = A @ B

    max_err = np.max(np.abs(C_dist - C_ref))
    print(f"[Coordinator] Max greška: {max_err:.6e}")
    print(f"[Coordinator] Tačnost OK: {max_err < 1e-4}")

if __name__ == "__main__":
    asyncio.run(main())