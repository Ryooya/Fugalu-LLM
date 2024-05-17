import time
import torch
from torch import cuda,bfloat16
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

start = time.time()

# モデルの保存場所
MODEL_SAVE_DIR = "model_save"


#model_name = "Fugaku-LLM/Fugaku-LLM-13B"
model_name = "Fugaku-LLM/Fugaku-LLM-13B-instruct"
#model_name = "Fugaku-LLM/Fugaku-LLM-13B-instruct-gguf"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_kwargs={"load_in_8bit": True},
    use_fast=False,
    cache_dir=MODEL_SAVE_DIR)

quant_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    cache_dir=MODEL_SAVE_DIR
    )

#model = torch.compile(model)

system_example = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
instruction_example = "論理的な書き方で論議を評価するための5つの主要な原則を説明してください。","リスト化した原則を使用して、論議を評価するための具体的な証拠について議論し、その証拠が論議を弱めるか強化するかを説明してください。"
#instruction_example = "スーパーコンピュータ「富岳」の名前の由来を教えてください。"

prompt = f"{system_example}\n\n### 指示:\n{instruction_example}\n\n### 応答:\n"

input_ids = tokenizer.encode(prompt,
                             add_special_tokens=False,
                             return_tensors="pt")
tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=2000,
    do_sample=True,
    temperature=0.1,
    top_p=1.0,
    repetition_penalty=1.0,
    top_k=0
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)
print("経過時間:", time.time() - start)
