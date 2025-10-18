from pprint import pprint

import requests


def invoke_predict_one():
    payload = {
        "s3_path": "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0026.JPG"
    }
    resp = requests.post("http://localhost:8000/predict", json=payload)
    assert resp.status_code == 200

    pprint(resp.json())


invoke_predict_one()