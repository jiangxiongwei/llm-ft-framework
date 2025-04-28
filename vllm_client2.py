import openai
import json
from transformers import AutoTokenizer

# 配置
server_url = "http://127.0.0.1:8000/v1"  # 替换为服务端 IP
api_key = "EMPTY"
model_name = "Qwen2.5-7B-Instruct-Merged"

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    base_url=server_url,
    api_key=api_key
)

# 加载分词器以调试输入
tokenizer = AutoTokenizer.from_pretrained("/home/linux/llm/Qwen2.5-7B-Instruct")

# 测试用例
test_cases = [
    {
        "prompt": "苏州现在气温多少",
        "expected_function": "get_weather_by_location",
        "expected_args": {"location": "苏州"}
    },
    {
        "prompt": "我想看看产品列表",
        "expected_function": "get_xsea_product_list",
        "expected_args": {}
    },
    {
        "prompt": "你喜欢猫吗？",
        "expected_function": None,
        "expected_args": None
    },
    {
        "prompt": "苏州天气怎么样？",
        "expected_function": "get_weather_by_location",
        "expected_args": {"location": "苏州"}
    },
    {
        "prompt": "查看产品",
        "expected_function": "get_xsea_product_list",
        "expected_args": {}
    },
    {
        "prompt": "苏州有什么景点？",
        "expected_function": None,
        "expected_args": None
    },
    {
        "prompt": "北京气温",
        "expected_function": "get_weather_by_location",
        "expected_args": {"location": "北京"}
    }
]

# 发送请求并验证
tp, tn, fp, fn = 0, 0, 0, 0
for case in test_cases:
    print(f"\n测试用例: {case['prompt']}")
    try:
        # 调试输入提示
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": case["prompt"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"输入提示:\n{prompt}")

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=128,
            # temperature=0.01,
            # top_p=1.0,
            # top_k=-1
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
                print("结果: 正确 (TP)")
                tp += 1
            else:
                print(f"结果: 错误，预期 function={case['expected_function']}, args={case['expected_args']} (FP)")
                fp += 1
        else:
            print(f"模型输出: 无 function call，内容={completion.content}")
            if case["expected_function"] is None:
                print("结果: 正确 (TN)")
                tn += 1
            else:
                print(f"结果: 错误，预期 function={case['expected_function']}, args={case['expected_args']} (FN)")
                fn += 1
                
    except Exception as e:
        print(f"请求失败: {e}")

# 统计指标
total = tp + tn + fp + fn
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"\n评估结果:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
