# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import time

base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"

def load_qwen_model():
    # 初始化 LLM
    llm = LLM(
        model=base_model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )
    return llm

def create_sampling_params():
    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=1024,
        stop_token_ids=[151643]
    )
    return sampling_params

def generate_response(llm, sampling_params, messages, tools):
    # 使用 llm.chat 带上 tools 参数
    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        tools=tools
    )
    # 提取生成的文本
    generated_text = outputs[0].outputs[0].text
    return generated_text

def main():
    # 加载模型
    print("Loading Qwen2.5-7B-Instruct model...")
    start_time = time.time()
    llm = load_qwen_model()
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # 创建采样参数
    sampling_params = create_sampling_params()
    
    # 定义 tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather_by_location",
                "description": "Get the current weather for a specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or location to get weather for."
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # 示例对话（使用 messages 格式）
    messages = [
        {
            "role": "system",
            "content": """You are Qwen, a helpful assistant. For queries requiring tools, return a tool call in <tool_call>{"name": "<function>", "arguments": <args>}</tool_call> format, followed by a brief text response."""
        },
        {
            "role": "user",
            "content": "苏州现在气温多少"
        }
    ]
    
    # 生成响应
    print("Generating response...")
    start_time = time.time()
    response = generate_response(llm, sampling_params, messages, tools)
    print(f"Response generated in {time.time() - start_time:.2f} seconds")
    
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main()