import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 路径配置
base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
lora_model_path = "/home/linux/source/experiment/outputs_size_1000/lora_model"
merged_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct-Merged"

# 加载基础模型
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
) 

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

# 加载 LoRA 适配器
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    model,
    lora_model_path,
    is_trainable=False
)

# 合并 LoRA 权重到基础模型
print("Merging LoRA weights...")
model = model.merge_and_unload()

# 保存合并后的模型
print(f"Saving merged model to {merged_model_path}...")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("Merged model saved successfully.")