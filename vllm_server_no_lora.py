# vllm_server_no_lora.py
# SPDX-License-Identifier: Apache-2.0

import uvloop
import os
from vllm import LLM
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser


# 设置环境
cli_env_setup()  # 模仿 CLI 的环境设置

# 配置
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
host = "0.0.0.0"
port = 8000
gpu_memory_utilization = 0.6  # 降低到 60%

# 设置环境变量减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 初始化 vLLM 模型
llm = LLM(
    model=base_model_path,
    gpu_memory_utilization=gpu_memory_utilization,
    max_model_len=256,  # 降低上下文长度
    trust_remote_code=True,
)

# 使用 cli_args 解析器获取默认参数
parser = make_arg_parser(FlexibleArgumentParser(description="vLLM CLI"))
args = parser.parse_args([
    "--model", base_model_path,
    "--host", host,
    "--port", str(port),
    "--max-model-len", "256",
    "--gpu-memory-utilization", str(gpu_memory_utilization),
    "--quantization", "bitsandbytes",
    "--trust-remote-code",
    "--tool-call-parser", "hermes",  # 尝试 hermes，若失败切换到 mistral
    "--enable-auto-tool-choice",
    "--log-level", "DEBUG",
    "--served-model-name", "Qwen2.5-7B-Instruct",
    "--enforce-eager",
    "--max-num-seqs", "16",
    "--dtype", "float16"
])

# 验证参数
validate_parsed_serve_args(args)

# 启动 OpenAI 兼容 API 服务器
print(f"Starting OpenAI-compatible API server at http://{host}:{port}")
uvloop.run(run_server(args, llm=llm))