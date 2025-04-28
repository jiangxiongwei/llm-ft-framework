from vllm import LLM, SamplingParams
import time


base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
def load_qwen_model():
    # 模型名称或路径
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 初始化 LLM
    llm = LLM(
        model=base_model_path,
        trust_remote_code=True,  # Qwen 需要这个参数
        tensor_parallel_size=1,   # 根据你的GPU数量调整
        gpu_memory_utilization=0.9,  # GPU内存利用率
        max_model_len=4096       # 最大模型长度
    )

    
    return llm

def create_sampling_params():
    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=1024,
        stop_token_ids=[151643],  # Qwen2的特殊结束token
    )
    
    return sampling_params

def generate_response(llm, sampling_params, prompt):
    # 生成响应
    outputs = llm.generate([prompt], sampling_params)
    
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
    
    # 示例对话
    prompt = """<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
请介绍一下量子计算的基本原理<|im_end|>
<|im_start|>assistant
"""
    
    # 生成响应
    print("Generating response...")
    start_time = time.time()
    response = generate_response(llm, sampling_params, prompt)
    print(f"Response generated in {time.time() - start_time:.2f} seconds")
    
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main()