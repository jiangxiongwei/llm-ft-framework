import os
import argparse
import uvloop
from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.tool_parsers import MistralToolParser
from vllm.entrypoints.openai.tool_parsers import hermes_tool_parser
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
# 配置
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
lora_model_path = "outputs_size_1000/lora_model"
host = "0.0.0.0"
port = 8000
gpu_memory_utilization = 0.9

# 初始化 vLLM 模型
llm = LLM(
    model=base_model_path,
    enable_lora=True,
    gpu_memory_utilization=gpu_memory_utilization,
    max_model_len=512,
    max_lora_rank=64,
    quantization="bitsandbytes",
    trust_remote_code=True
)

# 定义 LoRA 请求
lora_request = LoRARequest(
    lora_int_id=1,
    lora_name="rca_lora",
    lora_path=lora_model_path
)

# 构造 args 对象
args = argparse.Namespace(
    host=host,
    port=port,
    model=base_model_path,
    tokenizer=base_model_path,
    lora_modules=[f"{lora_request.lora_name}={lora_model_path}"],
    max_model_len=512,
    max_lora_rank=64,
    gpu_memory_utilization=gpu_memory_utilization,
    quantization="bitsandbytes",
    trust_remote_code=True,
    allow_credentials=True,
    log_level="INFO",
    disable_log_requests=False,
    max_log_len=None,
    uvicorn_log_level="info",
    response_role="assistant",
    chat_template=None,
    api_key=None,
    ssl_certfile=None,
    ssl_keyfile=None,
    ssl_ca_certs=None,
    ssl_cert_reqs="CERT_NONE",
    root_path="",
    middleware=[],
    disable_async=False,
    tool_parser_plugin=None,
    served_model_name="Qwen2.5-7B-Instruct"  # 显式设置服务模型名称
)

# 启动 OpenAI 兼容 API 服务器
print(f"Starting OpenAI-compatible API server at http://{host}:{port}")
uvloop.run(run_server(args, llm=llm, default_lora_requests=[lora_request]))