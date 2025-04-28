from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import uvicorn
import os
import json
import time
from typing import List, Optional, Dict, Any

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
HOST = "0.0.0.0"
PORT = 8888

# 定义请求和响应模型
class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: str
    function: Function

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict

# 初始化 FastAPI
app = FastAPI(title="Qwen2.5-7B-Instruct API")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 vLLM 引擎
engine_args = AsyncEngineArgs(
    model=base_model_path,
    trust_remote_code=True,
    tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
    gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
    max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

def create_qwen_chat_prompt(messages, tools=None):
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message.get("content", "")
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    prompt += f"<|im_start|>assistant\n{tool_call['function']['name']}({tool_call['function']['arguments']})<|im_end|>\n"
            else:
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            prompt += f"<|im_start|>tool\n{content}<|im_end|>\n"
    
    # 添加工具定义
    if tools:
        tool_prompt = "<|im_start|>tools\n"
        for tool in tools:
            tool_prompt += f"{tool.function.name}: {tool.function.description}\n"
            tool_prompt += f"parameters: {json.dumps(tool.function.parameters, ensure_ascii=False)}\n"
        tool_prompt += "<|im_end|>\n"
        prompt = tool_prompt + prompt
    
    # 添加 assistant 前缀以指示模型开始生成
    prompt += "<|im_start|>assistant\n"
    return prompt

def parse_tool_calls(text: str) -> List[ToolCall]:
    # 简单解析工具调用，实际应用中需要更健壮的解析逻辑
    tool_calls = []
    if "(" in text and ")" in text:
        function_name = text.split("(")[0].strip()
        arguments_str = text.split("(")[1].split(")")[0].strip()
        try:
            arguments = json.loads(arguments_str)
        except:
            arguments = arguments_str
        
        tool_calls.append({
            "id": f"call_{random_uuid()}",
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(arguments, ensure_ascii=False)
            }
        })
    return tool_calls

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI 兼容的聊天补全接口"""
    # if request.model != MODEL_NAME:
    #     raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
    
    # 创建 Qwen 格式的提示
    prompt = create_qwen_chat_prompt(request.messages, request.tools)
    
    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop_token_ids=[151643],  # Qwen2的特殊结束token
    )
    
    # 生成请求ID
    request_id = random_uuid()
    
    # 开始生成
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    # 非流式响应
    if not request.stream:
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if not final_output:
            raise HTTPException(status_code=500, detail="Failed to generate output")
        
        # 提取生成的文本
        generated_text = final_output.outputs[0].text
        
        # 尝试解析工具调用
        tool_calls = parse_tool_calls(generated_text)
        
        # 构造 OpenAI 兼容的响应
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=base_model_path,
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=None if tool_calls else generated_text,
                    tool_calls=[ToolCall(**tc) for tc in tool_calls] if tool_calls else None
                ),
                finish_reason="tool_calls" if tool_calls else "stop"
            )],
            usage={
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
            }
        )
    else:
        # 流式响应实现可以在这里添加
        raise HTTPException(status_code=501, detail="Streaming not implemented yet")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model": MODEL_NAME}

if __name__ == "__main__":
    print(f"Starting Qwen2.5-7B-Instruct API server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)