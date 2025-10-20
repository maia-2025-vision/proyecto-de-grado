from pprint import pprint

import requests


def invoke_predict_many() -> None:
    payloads = [
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0027.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0028.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0029.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0030.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0031.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0032.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0033.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0027.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0028.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0029.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0030.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0031.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0032.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0033.JPG",
        "s3://cow-detect-maia/andresalia/2025-04-01/DJI_0034.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0027.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0028.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0029.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0030.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0031.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0032.JPG",
        "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0033.JPG",
    ]
    for i, payload in enumerate(payloads):
        resp = requests.post("http://localhost:8000/predict", json={"s3_path": payload})
        assert resp.status_code == 200, f"{resp.text}"
        print(f"{i + 1} / {len(payloads)}")


def invoke_predict_one() -> None:
    payload = {"s3_path": "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0026.JPG"}
    resp = requests.post("http://localhost:8000/predict", json=payload)
    if resp.status_code != 200:
        print(f"ERROR!\n{resp.text}")
        return

    pprint(resp.json())


invoke_predict_many()
# invoke_predict_one()
