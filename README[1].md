# Adaptive SLM–PEFT Deployment Advisor

A benchmark-driven recommendation dashboard for selecting
Small Language Models, PEFT methods, and deployment configurations.

## Features
- GPU VRAM constraint filtering
- Latency and quality requirements
- Weighted deployment recommendation
- Ranked feasible alternatives
- Benchmark comparison

## Run locally

Install dependencies:

    pip install -r requirements.txt

Launch:

    streamlit run app.py

## Dataset
The dashboard uses the measured multi-model deployment benchmark
stored in multi_model_deployment_benchmark.csv.
