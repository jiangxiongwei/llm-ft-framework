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
lora_model_path = "outputs/lora_model"
validation_file = "validation_set.jsonl"
log_file = "validate.log"
errors_file = "errors.json"

valid_functions = ["get_xsea_product_list", "get_weather_by_location"]

def log_to_file(message, file=log_file):
    with open(file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_validation_set(filename):
    validation_set = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                validation_set.append(sample)
        log_to_file(f"Loaded validation set: {len(validation_set)} samples")
        log_to_file(f"First 5 samples: {[{'prompt': v['prompt'].split('<|im_start|>user\n')[1].split('<|im_end>')[0], 'expected': v['expected']} for v in validation_set[:5]]}")
        return validation_set
    except Exception as e:
        log_to_file(f"Error loading validation set: {e}")
        raise

def post_process_output(output):
    try:
        start = output.find("<tool_call>")
        end = output.find("</tool_call>")
        if start != -1 and end != -1:
            tool_call_str = output[start+11:end].strip()
            # 清理非法字符
            tool_call_str = tool_call_str.replace('\n', '').replace('\r', '').replace('\t', '').strip('\'"')
            # 规范化空格
            tool_call_str = ' '.join(tool_call_str.split())
            # 确保 JSON 格式
            if not tool_call_str.startswith('{') or not tool_call_str.endswith('}'):
                log_to_file(f"Invalid JSON format: {tool_call_str}")
                return output.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            tool_call = json.loads(tool_call_str)
            if tool_call["name"] not in valid_functions:
                return f"抱歉，无法识别 '{tool_call['name']}' 对应的功能。<|im_end><|endoftext|>"
            return f"<tool_call>{tool_call_str}</tool_call><|im_end><|endoftext|>"
        # 清理非 <tool_call> 输出
        return output.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    except json.JSONDecodeError as e:
        log_to_file(f"JSON decode error: {e}, Output: {output}")
        return output.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    except Exception as e:
        log_to_file(f"Post-process error: {e}, Output: {output}")
        return output.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

def evaluate_function_call(model, tokenizer, validation_set):
    model.eval()
    correct = 0
    total_function_calls = sum(1 for v in validation_set if "<tool_call>" in v["expected"])
    false_positives = 0
    total_negatives = sum(1 for v in validation_set if "<tool_call>" not in v["expected"])
    errors = []

    for i, val in enumerate(validation_set):
        try:
            inputs = tokenizer(val["prompt"], return_tensors="pt").to("cuda:0")
            outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            decoded_output = post_process_output(decoded_output)
            # 记录样本
            sample_log = f"Sample {i}: Prompt={val['prompt'].split('<|im_start|>user\n')[1].split('<|im_end>')[0]}\nOutput={decoded_output}\nExpected={val['expected']}\n"
            log_to_file(sample_log)
            print(sample_log)

            start = decoded_output.find("<tool_call>")
            end = decoded_output.find("</tool_call>")
            expected_start = val["expected"].find("<tool_call>")
            expected_end = val["expected"].find("</tool_call>")
            
            if start != -1 and end != -1:  # 模型输出 <tool_call>
                tool_call_str = decoded_output[start+11:end].strip()
                try:
                    tool_call = json.loads(tool_call_str)
                    tool_call_normalized = json.loads(json.dumps(tool_call, sort_keys=True, ensure_ascii=False))
                    if expected_start == -1:  # 负样本
                        false_positives += 1
                        error = f"False Positive: Sample {i}, Got {tool_call_str}, Expected {val['expected']}"
                        errors.append(error)
                        log_to_file(error)
                        print(error)
                    else:  # 正样本
                        expected_tool_call_str = val["expected"][expected_start+11:expected_end].strip()
                        expected_tool_call = json.loads(expected_tool_call_str)
                        expected_normalized = json.loads(json.dumps(expected_tool_call, sort_keys=True, ensure_ascii=False))
                        if tool_call_normalized == expected_normalized:
                            correct += 1
                        else:
                            error = f"Incorrect: Sample {i}, Got {tool_call_str}, Expected {expected_tool_call_str}"
                            errors.append(error)
                            log_to_file(error)
                            print(error)
                            # 记录 JSON 差异
                            diff = list(difflib.ndiff([tool_call_str], [expected_tool_call_str]))
                            log_to_file(f"JSON diff: {diff}")
                except json.JSONDecodeError as e:
                    error = f"JSON decode error: Sample {i}, Error: {e}, Output: {decoded_output}, Expected: {val['expected']}"
                    errors.append(error)
                    log_to_file(error)
                    print(error)
                    if expected_start == -1:
                        if val["expected"].strip() in decoded_output.strip() or decoded_output.strip().startswith(val["expected"].strip().split()[0]):
                            correct += 1
                        else:
                            error = f"Incorrect negative (post-JSON error): Sample {i}, Got {decoded_output}, Expected {val['expected']}"
                            errors.append(error)
                            log_to_file(error)
                            print(error)
            else:  # 模型无 <tool_call>
                if expected_start == -1:  # 负样本
                    if val["expected"].strip() in decoded_output.strip() or decoded_output.strip().startswith(val["expected"].strip().split()[0]):
                        correct += 1
                    else:
                        error = f"Incorrect negative: Sample {i}, Got {decoded_output}, Expected {val['expected']}"
                        errors.append(error)
                        log_to_file(error)
                        print(error)
                        # 记录文本差异
                        diff = list(difflib.ndiff([decoded_output], [val["expected"]]))
                        log_to_file(f"Text diff: {diff}")
                else:  # 正样本
                    error = f"Missed tool call: Sample {i}, Got {decoded_output}, Expected {val['expected']}"
                    errors.append(error)
                    log_to_file(error)
                    print(error)
        except Exception as e:
            error = f"Eval error: Sample {i}, Error: {e}, Output: {decoded_output}, Expected: {val['expected']}"
            errors.append(error)
            log_to_file(error)
            print(error)
            if expected_start == -1:
                if val["expected"].strip() in decoded_output.strip() or decoded_output.strip().startswith(val["expected"].strip().split()[0]):
                    correct += 1
                else:
                    error = f"Incorrect negative (post-eval error): Sample {i}, Got {decoded_output}, Expected {val['expected']}"
                    errors.append(error)
                    log_to_file(error)
                    print(error)

    accuracy = correct / len(validation_set)
    recall = correct / total_function_calls if total_function_calls > 0 else 0
    fpr = false_positives / total_negatives if total_negatives > 0 else 0
    final_log = f"\nFinal: Accuracy={accuracy:.4f}, Recall={recall:.4f}, FPR={fpr:.4f}, Correct={correct}, Total={len(validation_set)}, Total Function Calls={total_function_calls}, False Positives={false_positives}, Total Negatives={total_negatives}"
    log_to_file(final_log)
    print(final_log)

    # 保存错误汇总
    with open(errors_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    log_to_file(f"Saved error summary to {errors_file}")
    error_summary = f"\nError Summary ({len(errors)} errors):\n" + "\n".join(errors)
    log_to_file(error_summary)
    print(error_summary)

    return accuracy, recall, fpr

def main():
    # 清空日志文件
    if os.path.exists(log_file):
        os.remove(log_file)
    if os.path.exists(errors_file):
        os.remove(errors_file)

    # 加载模型
    log_to_file("Loading base model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=1024,
            dtype=torch.float16,
            load_in_4bit=False,
            device_map={"": 0},
            trust_remote_code=True
        )
        log_to_file("Base model loaded successfully")
        
        # 加载 LoRA 适配器
        model = PeftModel.from_pretrained(
            model,
            lora_model_path,
            device_map={"": 0},
            is_trainable=False
        )
        log_to_file("LoRA adapter loaded successfully")
        
        # 检查设备
        for name, param in model.named_parameters():
            log_to_file(f"Parameter {name} is on device: {param.device}")
            break
    except Exception as e:
        log_to_file(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        exit(1)

    # 启用推理模式
    FastLanguageModel.for_inference(model)

    # 测试用例
    test_prompts = [
        "昆山天气如何",
        "你喜欢购物吗？",
        "我想看看有哪些脚本",
        "我想看看有哪些计划",
        "我想看看有哪些目标",
        "我想看看有哪些书",
        "北京气温多少",
        "上海天气预报",
        "广州天气怎么样"
    ]
    log_to_file("\n=== Test Results ===")
    for prompt in test_prompts:
        try:
            input_text = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant capable of executing function calls. Available functions: get_xsea_product_list(), get_weather_by_location(location). Respond with a <tool_call> containing a JSON object with 'name' and 'arguments'. For non-function queries, respond with plain text.<|im_end>\n<|im_start|>user\n{prompt}<|im_end>\n<|im_start|>assistant"
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            test_log = f"Input: {prompt}\nOutput: {decoded_output}\n"
            log_to_file(test_log)
            print(test_log)
        except Exception as e:
            error = f"Error processing prompt '{prompt}': {e}"
            log_to_file(error)
            print(error)

    # 加载验证集
    log_to_file(f"\nLoading validation set from {validation_file}...")
    validation_set = load_validation_set(validation_file)

    # 验证
    accuracy, recall, fpr = evaluate_function_call(model, tokenizer, validation_set)
    log_to_file(f"Results: Accuracy={accuracy:.4f}, Recall={recall:.4f}, FPR={fpr:.4f}")

    # 清理
    del model, tokenizer
    torch.cuda.empty_cache()
    log_to_file(f"Cleared PyTorch CUDA cache, allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GiB, reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")

if __name__ == "__main__":
    main()