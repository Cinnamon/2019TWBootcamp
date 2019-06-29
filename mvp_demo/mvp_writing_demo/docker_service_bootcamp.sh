#!/bin/bash

set -m

tensorflow_model_server --rest_api_port=8501 --port=8500 \
    --model_name=bootcamp_demo \
    --model_base_path=/models/bootcamp_demo &

python3 ./flask_app/app_bootcamp.py
