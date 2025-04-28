import torch
import os
import json
import difflib
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

base_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"
lora_model_path = "outputs_size_1000/lora_model"

# 加载模型
print("Loading base model...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=False,
        device_map={"": 0},
        trust_remote_code=True
    )
    print("Base model loaded successfully")
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(
        model,
        lora_model_path,
        device_map={"": 0},
        is_trainable=False
    )
    print("LoRA adapter loaded successfully")
    
    # 检查设备
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")
        break
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 启用推理模式
FastLanguageModel.for_inference(model)



# 测试输入
prompt = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
<|im_end>
<|im_start|>user
苏州现在气温多少
<|im_end>
<|im_start|>assistant"""

prompt2 = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
<|im_end>
<|im_start|>user
你喜欢猫吗
<|im_end>
<|im_start|>assistant"""

inputs = tokenizer(prompt2, return_tensors="pt").to("cuda:0")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id
            )
response2 = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("模型输出:", response2)