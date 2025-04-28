import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 路径配置
model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"

# 加载原始模型
print("Loading Qwen2.5-7B-Instruct model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 启用推理模式
model.eval()

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
    },
    {
        "type": "function",
        "function": {
            "name": "get_xsea_product_list",
            "description": "Retrieve the list of available products.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# 测试用例
test_case = {
    "prompt": "苏州现在气温多少",
    "tools": tools
}

# 构造输入
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": test_case["prompt"]}
]
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True
)

# 打印输入提示
print(f"\n输入提示:\n{prompt}")

# 生成输出
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer
)

# 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n模型输出:\n{response}")