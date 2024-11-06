import argparse
import base64
import json
import os
import pprint
from pathlib import Path
from typing import Any, Dict

import requests

TEST_CASES = ["request_match.json", "request_mismatch.json"]


def transform_test_data_into_request_data(data: Dict[str, Any]):
    video_idx = -1
    for i, input in enumerate(data["inputs"]):
        if input["name"] == "video":
            video_idx = i
            break

    data["inputs"][video_idx]["data"] = [
        read_video_b64(video_filename)
        for video_filename in data["inputs"][video_idx]["data"]
    ]


def read_video_b64(video_filename: str) -> str:
    dir = Path(__file__).parent.resolve()
    with open(dir / video_filename, "rb") as video_file:
        video_data = video_file.read()
        return base64.b64encode(video_data).decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="local", description="Test local Triton server endpoint"
    )
    parser.add_argument("-o", "--output-dir", default=os.getcwd())
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    for test_case in TEST_CASES:
        print(f"Sending {test_case}...")
        with open(dir_path / test_case) as data_file:
            data = json.loads(data_file.read())

        transform_test_data_into_request_data(data)

        response = requests.post(
            "http://localhost:8000/v2/models/detector/versions/1/infer",
            data=json.dumps(data),
        )
        pprint.pprint(response.json())
