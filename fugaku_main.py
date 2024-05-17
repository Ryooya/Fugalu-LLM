import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

start = time.time()

MODEL_SAVE_DIR = "model_save"

model_name = "Fugaku-LLM/Fugaku-LLM-13B-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=MODEL_SAVE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=MODEL_SAVE_DIR
    )

model.eval()

system_example = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
instruction_example = "スーパーコンピュータ「富岳」の名前の由来を教えてください。"

prompt = f"{system_example}\n\n### 指示:\n{instruction_example}\n\n### 応答:\n"

input_ids = tokenizer.encode(prompt,
                             add_special_tokens=False,
                             return_tensors="pt")
tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
    do_sample=True,
    temperature=0.1,
    top_p=1.0,
    repetition_penalty=1.0,
    top_k=0
)
out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)
print("経過時間:", time.time() - start)
