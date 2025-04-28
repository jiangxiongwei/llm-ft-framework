#!/bin/bash

vllm serve /home/linux/llm/Qwen2.5-7B-Instruct   \
    --host 0.0.0.0 --port 8000  \
    --max-model-len 1024  \
    --dtype float16  \
    --gpu-memory-utilization 0.9  \
    --trust-remote-code --served-model-name Qwen2.5-7B-Instruct \ 
    --enable-auto-tool-choice \
    --tool-call-parser hermes
