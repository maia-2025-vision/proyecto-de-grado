"""Script to test different endpoint end to end"""

from pathlib import Path
from pprint import pprint
from time import perf_counter

import requests
from loguru import logger
from typer import Typer

API_BASE_URL = "http://localhost:8000"

cli = Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)

many_uris = [
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


@cli.command("predict-one", help="Calle /predict endpoint once")
def invoke_predict_one() -> None:
    payload = {"s3_path": "s3://cow-detect-maia/andresalia/2024-10-10/DJI_0026.JPG"}
    resp = requests.post(f"{API_BASE_URL}/predict", json=payload)
    if resp.status_code != 200:
        print(f"ERROR!\n{resp.text}")
        return

    pprint(resp.json())


DEFAULT_FILE = Path(
    # Big image (~3600 * 5400) takes about 40 seconds on CPU
    # "data/train/006b4661847b82acfb2b6a3e3677f4ae63f1dd5c.JPG"
    # Individual patch: 512 x 512, takes 1.2 seconds on CPU
    "data/patches-512-ol-160-m0.3/train/006b4661847b82acfb2b6a3e3677f4ae63f1dd5c_101.JPG"
)


@cli.command("pred-on-upl", help="Call /predict-on-uploaded endpoint once")
def invoke_predict_on_upload(file_path: Path = DEFAULT_FILE) -> None:
    logger.info(f"Uploading image file: {file_path}")

    tic = perf_counter()
    with file_path.open("rb") as f:
        # The 'files' dictionary is the standard way to send 'multipart/form-data'
        files = {"file": (file_path.name, f)}
        # Send the POST request
        resp = requests.post(f"{API_BASE_URL}/predict-on-upload", files=files)
    elapsed = perf_counter() - tic
    logger.info(f"Prediction took: {elapsed:.3f} seconds.")

    if resp.status_code != 200:
        print(f"ERROR!\n{resp.text}")
        return

    pprint(resp.json())


@cli.command("predict-one-mult", help="Call /predict multiple times")
def invoke_predict_one_multiple_times() -> None:
    for i, s3_uri in enumerate(many_uris):
        resp = requests.post(f"{API_BASE_URL}/predict", json={"s3_path": s3_uri})
        assert resp.status_code == 200, f"{resp.text}"
        print(f"{i + 1} / {len(many_uris)}")


@cli.command("predict-many", help="Call /predict-many once")
def invoke_predict_many_once(batch_size: int = 4) -> None:
    payload = {"urls": many_uris[:batch_size]}
    tic = perf_counter()
    resp = requests.post(f"{API_BASE_URL}/predict-many", json=payload)
    assert resp.status_code == 200, f"{resp.text}"
    elapsed = perf_counter() - tic
    logger.info(f"Prediction took: {elapsed:.3f} seconds.")
    # pprint(resp.json())


if __name__ == "__main__":
    cli()
