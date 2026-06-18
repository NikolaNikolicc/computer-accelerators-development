#!/usr/bin/env python3
import time
import numpy as np
from azure.iot.device import IoTHubDeviceClient, MethodResponse

DEVICE_CONNECTION_STRING = "HostName=<your-hub>.azure-devices.net;DeviceId=wsn-node-01;SharedAccessKey=<DEVICE_KEY>"

def local_matmul(A_chunk, B):
    A = np.array(A_chunk, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    C = A @ B
    return C.tolist()

def method_request_handler(method_request):
    if method_request.name != "multiply":
        response = MethodResponse.create_from_method_request(
            method_request,
            status=404,
            payload={"error": f"Nepoznat metod: {method_request.name}"}
        )
        return response

    payload = method_request.payload
    task_id = payload["task_id"]
    row_start = payload["row_start"]
    row_end = payload["row_end"]
    A_chunk = payload["A_chunk"]
    B = payload["B"]

    print(f"[Worker] Primljen task {task_id}, redovi [{row_start}:{row_end}]")
    t0 = time.time()
    C_chunk = local_matmul(A_chunk, B)
    compute_time = time.time() - t0

    response_payload = {
        "task_id": task_id,
        "row_start": row_start,
        "row_end": row_end,
        "C_chunk": C_chunk,
        "compute_time_s": compute_time
    }

    response = MethodResponse.create_from_method_request(
        method_request,
        status=200,
        payload=response_payload
    )
    return response

def main():
    print("[Worker] Konekcija na Azure IoT Hub...")
    client = IoTHubDeviceClient.create_from_connection_string(DEVICE_CONNECTION_STRING)
    client.connect()
    print("[Worker] Povezan.")

    def on_method_received(method_request):
        response = method_request_handler(method_request)
        client.send_method_response(response)
        print("[Worker] MethodResponse poslat.")

    client.on_method_request_received = on_method_received

    try:
        print("[Worker] Čekam Direct Method pozive...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Worker] Zaustavljanje.")
    finally:
        client.shutdown()

if __name__ == "__main__":
    main()