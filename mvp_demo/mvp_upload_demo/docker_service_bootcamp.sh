#!/bin/bash

set -m

tensorflow_model_server --rest_api_port=8501 --port=8500 \
    --model_name=bootcamp_demo2 \
    --model_base_path=/models/bootcamp_demo2 &

python3 ./flask_app/app_bootcamp2.py
