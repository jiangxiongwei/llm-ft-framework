import openai
import json
# 配置
server_url = "http://127.0.0.1:8000/v1"  # 替换为服务端 IP
api_key = "EMPTY"  # vLLM 不需 API 密钥，占位符
# model_name = "Qwen2.5-7B-Instruct"
model_name = "Qwen2.5-7B-Instruct"
# 初始化 OpenAI 客户端
client = openai.OpenAI(
    base_url=server_url,
    api_key=api_key
)

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_location",
            "description": "Get the weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get weather for"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_xsea_product_list",
            "description": "Get the list of available products",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# 测试用例
test_cases = [
    {
        "prompt": "苏州现在气温多少",
        "expected_function": "get_weather_by_location",
        "expected_args": {"location": "苏州"}
    }
]

# 发送请求并验证
for case in test_cases:
    print(f"\n测试用例: {case['prompt']}")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Always provide a brief text response in addition to any tool calls to confirm the action"},
                {"role": "user", "content": case["prompt"]}
            ],
            tools=tools,
            tool_choice="auto",
            # max_tokens=128,
            # temperature=0.7
        )
        
        # 解析响应
        completion = response.choices[0].message
        if completion.tool_calls:
            tool_call = completion.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"模型输出: function={function_name}, args={function_args}")
            
            # 验证
            if case["expected_function"] == function_name and case["expected_args"] == function_args:
                print("结果: 正确")
            else:
                print(f"结果: 错误，预期 function={case['expected_function']}, args={case['expected_args']}")
        else:
            print(f"模型输出: 无 function call，内容={completion.content}")
            if case["expected_function"] is None:
                print("结果: 正确")
            else:
                print(f"结果: 错误，预期 function={case['expected_function']}, args={case['expected_args']}")
                
    except Exception as e:
        print(f"请求失败: {e}")
