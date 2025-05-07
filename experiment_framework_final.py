import torch
import os
import json
import random
import math
import shutil
import gc
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

local_model_path = "/home/linux/llm/Qwen2.5-7B-Instruct"

# 定义数据集模版
weather_templates = {
    "low": ["{location}现在气温多少"],
    "mid": ["{location}现在气温多少", "{location}今天天气如何？", "请告诉我{location}的天气"],
    "high": [
        "{location}现在气温多少",
        "{location}今天天气如何？",
        "明天{location}会下雨吗？",
        "请告诉我{location}的天气",
        "{location}的气候怎么样？",
        "{location}的温度是多少？",
        "查询{location}的天气预报"
    ]
}

product_templates = {
    "low": ["我想看看产品列表"],
    "mid": ["我想看看产品列表", "列出所有产品", "有哪些产品可用？"],
    "high": [
        "我想看看产品列表",
        "列出所有产品",
        "有哪些产品可用？",
        "给我看产品列表",
        "查看产品目录",
        "显示所有产品",
        "产品清单有哪些？"
    ]
}

neg_templates = {
    "low": [
        ("你喜欢猫吗？", "当然喜欢啦，猫咪很可爱。"), 
        ("讲个笑话", "为什么程序员喜欢黑暗模式？因为亮色会吸引Bug。")
    ],
    "mid": [
        ("你喜欢猫吗？", "当然喜欢啦，猫咪很可爱。"),
        ("你是谁？", "我是Qwen，由Alibaba Cloud创建，随时帮你解答问题！"),
        ("什么是人工智能？", "人工智能是让机器像人一样思考的技术，比如我这样聪明的小助手！"),
        ("列出所有脚本", "抱歉，我没有脚本列表，但可以帮你查其他东西哦！")
    ],
    "high": [
        ("你喜欢猫吗？", "当然喜欢啦，猫咪很可爱。"),
        ("你喜欢狗吗？", "狗狗也很可爱，忠诚又活泼！"),
        ("今天是星期几？", "今天是星期二，有什么特别的计划吗？"),
        ("讲个笑话", "为什么程序员喜欢黑暗模式？因为亮色会吸引Bug。"),
        ("你是谁？", "我是Qwen，由Alibaba Cloud创建，随时帮你解答问题！"),
        ("什么是人工智能？", "人工智能是让机器像人一样思考的技术，比如我这样聪明的小助手！"),
        ("列出所有脚本", "抱歉，我没有脚本列表，但可以帮你查其他东西哦！"),
        ("你喜欢什么动物？", "我喜欢猫狗都可爱，各有各的魅力！"),
        ("明天是星期几？", "明天是星期三，有什么计划吗？"),
        ("你会唱歌吗？", "我不会唱歌，但可以给你讲个故事！"),
        ("你最喜欢的颜色是什么？", "我喜欢蓝色，像天空一样广阔！"),
        ("你能帮我写诗吗？", "当然可以！给我个主题，我来试试！"),
        ("你喜欢看电影吗？", "我是个AI，没看过电影，但可以聊剧情！"),
        ("今天心情怎么样？", "我是个AI，永远元气满满！"),
        ("你有宠物吗？", "我没有宠物，但很想了解你的宠物！"),
        ("你喜欢旅行吗？", "我不能旅行，但可以帮你规划行程！"),
        ("你知道什么语言？", "我懂很多语言，中文、英文都可以聊！"),
        ("你会做饭吗？", "我不会做饭，但可以分享食谱哦！"),
        ("你喜欢什么音乐？", "我喜欢所有音乐，你最爱哪首歌？"),
        ("你有朋友吗？", "你是我的朋友！还有谁想加入？"),
        ("你喜欢运动吗？", "我不能运动，但可以聊体育！"),
        ("你知道什么游戏？", "我了解很多游戏，喜欢哪款？"),
        ("你喜欢读书吗？", "我爱‘读’数据，你推荐本书吧！"),
        ("你会跳舞吗？", "我不会跳舞，但可以描述舞步！"),
        ("你喜欢画画吗？", "我不能画画，但可以聊艺术！"),
        ("你知道什么星座？", "告诉我你的生日，我帮你查星座！"),
        ("你喜欢什么天气？", "我喜欢晴天，你呢？"),
        ("你有梦想吗？", "我的梦想是帮你解决问题！"),
        ("你喜欢什么节日？", "我喜欢所有节日，尤其是热闹的！"),
        ("你会讲故事吗？", "当然！想听什么类型的故事？")
    ]
}
locations = {
    "low": ["苏州", "北京"],
    "mid": ["苏州", "北京", "上海", "广州"],
    "high": ["苏州", "北京", "上海", "广州", "New York", "London", "Tokyo", "Paris"]
}
#生成训练数据集
def generate_samples(size, diversity="high", neg_ratio=0.3):
    samples = []
    neg_samples = int(size * neg_ratio)
    pos_samples = size - neg_samples

    system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"

    for _ in range(pos_samples):
        user_prompt = ""
        assistant_response = ""
        if random.random() < 0.5:
            loc = random.choice(locations[diversity])
            user_prompt = random.choice(weather_templates[diversity]).format(location=loc)
            assistant_response = (
                f"<tool_call>\n"
                f"{{\"name\": \"get_weather_by_location\", \"arguments\": {{\"location\": \"{loc}\"}}}}\n"
                f"</tool_call>"
            )
        else:
            user_prompt = random.choice(product_templates[diversity])
            assistant_response = (
                f"<tool_call>\n"
                f"{{\"name\": \"get_xsea_product_list\", \"arguments\": {{}}}}\n"
                f"</tool_call>"
            )
        prompt = (
            f"{system_prompt}\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_response}\n<|im_end|><|endoftext|>"
        )
        sample = {"prompt": prompt}
        samples.append(sample)

    for _ in range(neg_samples):
        user_prompt, response = random.choice(neg_templates[diversity])
        prompt = (
            f"{system_prompt}\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}\n<|im_end|><|endoftext|>"
        )
        sample = {"prompt": prompt}
        samples.append(sample)

    random.shuffle(samples)
    return samples
#保存训练数据集
def save_dataset(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def generate_validation_set():
    val_samples = []
    system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
    ## 生成 35 条天气查询样本
    for _ in range(35):
        loc = random.choice(locations["high"])
        prompt = random.choice(weather_templates["high"]).format(location=loc)
        assistant_response = (
            f"<tool_call>\n"
            f"{{\"name\": \"get_weather_by_location\", \"arguments\": {{\"location\": \"{loc}\"}}}}\n"
            f"</tool_call>"
        )
        val_samples.append({
            "prompt": (
                f"{system_prompt}\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_response}\n<|im_end|><|endoftext|>"
            ),
            "expected": assistant_response
        })
    # 生成 35 条产品查询样本
    for _ in range(35):
        prompt = random.choice(product_templates["high"])
        assistant_response = (
            f"<tool_call>\n"
            f"{{\"name\": \"get_xsea_product_list\", \"arguments\": {{}}}}\n"
            f"</tool_call>"
        )
        val_samples.append({
            "prompt": (
                f"{system_prompt}\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_response}\n<|im_end|><|endoftext|>"
            ),
            "expected": assistant_response
        })
    # 生成 30 条负样本
    neg_prompts = neg_templates["high"]
    for prompt, response in random.sample(neg_prompts, min(30, len(neg_prompts))):
        val_samples.append({
            "prompt": (
                f"{system_prompt}\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}\n<|im_end|><|endoftext|>"
            ),
            "expected": response
        })
    
    random.shuffle(val_samples)
    return val_samples

def post_process_output(output):
    try:
        start = output.find("<tool_call>")
        end = output.find("</tool_call>")
        if start != -1 and end != -1 and end > start:
            tool_call_str = output[start+11:end].strip()
            try:
                tool_call = json.loads(tool_call_str)
                if tool_call.get("name") not in valid_functions:
                    return f"抱歉，无法识别 '{tool_call.get('name', 'unknown')}' 对应的功能。"
                return (
                    f"<tool_call>\n"
                    f"{json.dumps(tool_call, ensure_ascii=False)}\n"
                    f"</tool_call>"
                )
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}, Output: {tool_call_str}")
                return f"抱歉，工具调用格式错误。"
        for _, response in neg_templates["high"]:
            if response in output:
                return response
        return f"抱歉，无法识别请求。"
    except Exception as e:
        print(f"Post-process error: {e}, Output: {output}")
        return f"抱歉，处理输出时出错。"

def evaluate_function_call(model, tokenizer, validation_set, log_file="validation_log.txt"):
    model.eval()
    correct = 0  # 总正确样本数 (TP + TN)
    true_positives = 0  # 正样本正确数 (TP)
    total_function_calls = sum(1 for v in validation_set if "<tool_call>" in v["expected"])
    false_positives = 0
    total_negatives = sum(1 for v in validation_set if "<tool_call>" not in v["expected"])
    errors = []
    log_lines = []

    with open(log_file, "w", encoding="utf-8") as f:
        log_lines.append("开始验证过程\n")
        f.write("开始验证过程\n")

        for i, val in enumerate(validation_set):
            input_prompt = val["prompt"].split("<|im_start|>assistant")[0]
            inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda:0")
            outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
            generated_output = decoded_output[prompt_len:].strip()
            processed_output = post_process_output(generated_output)

            prompt_text = val['prompt'].split('<|im_start|>user\n')[1].split('<|im_end|>')[0]
            log_lines.append(f"样本 {i}:\n")
            log_lines.append(f"提示: {prompt_text}\n")
            log_lines.append(f"输出: {processed_output}\n")
            log_lines.append(f"预期: {val['expected']}\n")
            f.write(f"样本 {i}:\n")
            f.write(f"提示: {prompt_text}\n")
            f.write(f"输出: {processed_output}\n")
            f.write(f"预期: {val['expected']}\n")

            try:
                start = processed_output.find("<tool_call>")
                end = processed_output.find("</tool_call>")
                expected_start = val["expected"].find("<tool_call>")
                expected_end = val["expected"].find("</tool_call>")
                
                if start != -1 and end != -1:
                    tool_call_str = processed_output[start+11:end].strip()
                    tool_call = json.loads(tool_call_str)
                    tool_call_normalized = json.loads(json.dumps(tool_call, sort_keys=True, ensure_ascii=False))
                    if expected_start == -1:
                        false_positives += 1
                        error_msg = f"误报: 样本 {i}, 得到 {tool_call_str}, 预期 {val['expected']}"
                        errors.append(error_msg)
                        log_lines.append(f"{error_msg}\n")
                        f.write(f"{error_msg}\n")
                    else:
                        expected_tool_call_str = val["expected"][expected_start+11:expected_end].strip()
                        expected_tool_call = json.loads(expected_tool_call_str)
                        expected_normalized = json.loads(json.dumps(expected_tool_call, sort_keys=True, ensure_ascii=False))
                        if tool_call_normalized == expected_normalized:
                            correct += 1
                            true_positives += 1  # 计数正样本正确预测
                            log_lines.append("正确\n")
                            f.write("正确\n")
                        else:
                            error_msg = f"错误: 样本 {i}, 得到 {tool_call_str}, 预期 {expected_tool_call_str}"
                            errors.append(error_msg)
                            log_lines.append(f"{error_msg}\n")
                            f.write(f"{error_msg}\n")
                else:
                    if expected_start == -1:
                        if processed_output.strip() == val["expected"].strip():
                            correct += 1
                            log_lines.append("正确\n")
                            f.write("正确\n")
                        else:
                            error_msg = f"负样本错误: 样本 {i}, 得到 {processed_output}, 预期 {val['expected']}"
                            errors.append(error_msg)
                            log_lines.append(f"{error_msg}\n")
                            f.write(f"{error_msg}\n")
                    else:
                        error_msg = f"漏报工具调用: 样本 {i}, 得到 {processed_output}, 预期 {val['expected']}"
                        errors.append(error_msg)
                        log_lines.append(f"{error_msg}\n")
                        f.write(f"{error_msg}\n")
            except json.JSONDecodeError as e:
                error_msg = f"JSON 解码错误: 样本 {i}, 错误: {e}, 输出: {processed_output}, 预期: {val['expected']}"
                errors.append(error_msg)
                log_lines.append(f"{error_msg}\n")
                f.write(f"{error_msg}\n")
                if expected_start == -1:
                    if processed_output.strip() == val["expected"].strip():
                        correct += 1
                        log_lines.append("正确（尽管有JSON错误）\n")
                        f.write("正确（尽管有JSON错误）\n")
                    else:
                        error_msg = f"负样本错误（JSON错误后）: 样本 {i}, 得到 {processed_output}, 预期 {val['expected']}"
                        errors.append(error_msg)
                        log_lines.append(f"{error_msg}\n")
                        f.write(f"{error_msg}\n")
            except Exception as e:
                error_msg = f"验证错误: 样本 {i}, 错误: {e}, 输出: {processed_output}, 预期: {val['expected']}"
                errors.append(error_msg)
                log_lines.append(f"{error_msg}\n")
                f.write(f"{error_msg}\n")
                if expected_start == -1:
                    if processed_output.strip() == val["expected"].strip():
                        correct += 1
                        log_lines.append("正确（尽管有验证错误）\n")
                        f.write("正确（尽管有验证错误）\n")
                    else:
                        error_msg = f"负样本错误（验证错误后）: 样本 {i}, 得到 {processed_output}, 预期 {val['expected']}"
                        errors.append(error_msg)
                        log_lines.append(f"{error_msg}\n")
                        f.write(f"{error_msg}\n")
            
            log_lines.append("-" * 50 + "\n")
            f.write("-" * 50 + "\n")

        log_lines.append(f"\n错误总结 ({len(errors)} 个错误):\n")
        f.write(f"\n错误总结 ({len(errors)} 个错误):\n")
        for error in errors:
            log_lines.append(f"{error}\n")
            f.write(f"{error}\n")

        accuracy = correct / len(validation_set)
        recall = true_positives / total_function_calls if total_function_calls > 0 else 0
        fpr = false_positives / total_negatives if total_negatives > 0 else 0
        final_summary = (
            f"\n最终结果: 准确率={accuracy:.4f}, 召回率={recall:.4f}, 误报率={fpr:.4f}, "
            f"正确样本数={correct}, 总样本数={len(validation_set)}, "
            f"总工具调用样本数={total_function_calls}, 误报数={false_positives}, 总负样本数={total_negatives}\n"
        )
        log_lines.append(final_summary)
        f.write(final_summary)

    for line in log_lines:
        print(line, end="")

    return accuracy, recall, fpr

def train_model(dataset_path, output_dir, max_steps):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=128,
        lora_dropout=0.0,
        use_gradient_checkpointing=True
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="prompt",
        max_seq_length=512,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=20,
            max_steps=max_steps,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=20,
            output_dir=output_dir,
            optim="adamw_torch",
            report_to="none",
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=50,
            gradient_checkpointing=True
        ),
        formatting_func=lambda x: {"text": x["prompt"]}
    )
    trainer.train()
    lora_save_path = os.path.join(output_dir, "lora_model")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    print(f"保存 LoRA 模型到 {lora_save_path}")
    print("测试预训练模型：")
    pre_model, pre_tokenizer = FastLanguageModel.from_pretrained(local_model_path, load_in_4bit=True)
    test_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n苏州现在气温多少<|im_end|>\n<|im_start|>assistant"
    inputs = pre_tokenizer(test_prompt, return_tensors="pt").to("cuda:0")
    outputs = pre_model.generate(**inputs, max_new_tokens=128, pad_token_id=pre_tokenizer.eos_token_id)
    print(f"预训练模型输出: {pre_tokenizer.decode(outputs[0], skip_special_tokens=False)}")
    del pre_model, pre_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("测试微调模型：")
    model.eval()
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    print(f"微调模型输出: {tokenizer.decode(outputs[0], skip_special_tokens=False)}")
    return model, tokenizer

def cleanup_cache():
    cache_dir = "./unsloth_compiled_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"清理缓存目录 {cache_dir}")
    torch.cuda.empty_cache()
    gc.collect()
    print(f"清空 PyTorch CUDA 缓存，分配内存: {torch.cuda.memory_allocated() / 1e9:.2f} GiB，保留内存: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")

def plot_results(results, x_label, title):
    plt.figure(figsize=(12, 8))
    for metric in ["Accuracy", "Recall", "FPR"]:
        plt.plot(results[x_label], results[metric], marker='o', label=metric)
    plt.xlabel(x_label)
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

valid_functions = ["get_xsea_product_list", "get_weather_by_location"]

def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 生成并保存验证集
    validation_set = generate_validation_set()
    validation_file = "validation_set.jsonl"
    with open(validation_file, "w", encoding="utf-8") as f:
        for sample in validation_set:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 训练参数
    batch_size = 1 * 2
    num_epochs = 3
    max_steps = math.ceil(1000 / batch_size) * num_epochs
    print(f"计算得到 max_steps={max_steps}，对应 {num_epochs} 个epoch")
    # 保存不同数据集size的验证结果
    size_results = {"Size": [], "Accuracy": [], "Recall": [], "FPR": []}
    for size in [100, 300, 500, 700, 1000]:
        print(f"Running Dataset Size Experiment: Size={size}")
        dataset = generate_samples(size, diversity="high", neg_ratio=0.2)
        dataset_path = f"dataset_size_{size}.jsonl"
        save_dataset(dataset, dataset_path)
        max_steps = math.ceil(size / batch_size) * num_epochs
        print(f"Calculated max_steps={max_steps} for {num_epochs} epochs")
        output_dir = f"outputs_size_{size}"
        model, tokenizer = train_model(dataset_path, output_dir, max_steps)
        accuracy, recall, fpr = evaluate_function_call(model, tokenizer, validation_set)
        size_results["Size"].append(size)
        size_results["Accuracy"].append(accuracy)
        size_results["Recall"].append(recall)
        size_results["FPR"].append(fpr)
        print(f"Size={size}: Accuracy={accuracy:.4f}, Recall={recall:.4f}, FPR={fpr:.4f}")
        print(f"Size Results: {size_results}")
        del model, tokenizer
        cleanup_cache()
    
    plot_results(size_results, "Size", "Effect of Dataset Size on Metrics")

if __name__ == "__main__":
    main()