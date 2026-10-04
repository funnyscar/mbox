import os
import torch
import torch.nn as nn
import tiktoken
import modal
import json
import math

app = modal.App("post_train")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "datasets")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("postmalone", create_if_missing=True)

@app.function(image=image, cpu=4, volumes={"/data": vol}, timeout=60 * 60)
def process_data(state_dict, total_steps=5000):
	from datasets import load_dataset

	ds = load_dataset("databricks/databricks-dolly-15k", split="train")

	def to_messages(ex):
		user = ex["instruction"]
		if ex["context"]:
			user += "\n\nContext:\n" + ex["context"]
		return {"messages": [
			{"role": "user", "content": user},
			{"role": "assistant", "content": ex["response"]},
		]}

	ds = ds.map(to_messages, remove_columns=ds.column_names)
	ds = ds.filter(lambda ex: len(ex["messages"][1]["content"]) > 0)   # drop empty answers

	split = ds.train_test_split(test_size=0.02, seed=42)
	split["train"].to_json("/data/train.jsonl")
	split["test"].to_json("/data/eval.jsonl")
	print(split)




@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=60 * 60)
def train():
	import torch
	import torch.nn as nn
	from datasets import load_dataset
	from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

	model_id = "Qwen/Qwen3-8B"  # or a smaller Qwen for testing

	tok = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=BitsAndBytesConfig(
			load_in_4bit=True, bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.bfloat16),
		device_map="auto",
	)
	def encode(ex, max_len=2048):
		prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
		full   = tok.apply_chat_template(ex["messages"], tokenize=False)
		p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
		ids   = tok(full,   add_special_tokens=False)["input_ids"][:max_len]
		labels = [-100] * len(p_ids) + ids[len(p_ids):]       # -100 = ignored in loss
		return {"input_ids": ids, "labels": labels[:len(ids)]}

	# JSONL with {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}
	ds = load_dataset("json", data_files="/data/train.jsonl", split="train").map(encode, remove_columns=["messages"])

	def collate(batch):
		L = max(len(b["input_ids"]) for b in batch)
		pad = lambda seq, v: seq + [v] * (L - len(seq))
		ids  = torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch])
		lab  = torch.tensor([pad(b["labels"], -100) for b in batch])
		mask = (torch.arange(L)[None] < torch.tensor([len(b["input_ids"]) for b in batch])[:, None]).long()
		return {"input_ids": ids, "labels": lab, "attention_mask": mask}

	dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)

	############### ABOVE UNCHANGED ################

	lora = LoraConfig(
		r=16, lora_alpha=32, lora_dropout=0.05,
		task_type="CAUSAL_LM",
	)

	class LoRALinear(nn.Module):
		def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
			super().__init__()
			self.base = base
			base.weight.requires_grad_(False) # freeze
			if base.bias is not None:
				base.bias.requires_grad_(False)

			self.A = nn.Parameter(torch.empty(r, base.in_features))
			self.B = nn.Parameter(torch.zeros(base.out_features, r))
			nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

			self.scale = alpha / r
			self.drop = nn.Dropout(dropout)

		def forward(self, x):
			x = self.scale * (self.drop(x) @ self.A.T @ self.B.T) # delta W
			x += self.base(x) # OLD linear weights
			return x

	def inject_lora(model, targets, **kw):
		for p in model.parameters():
			p.requires_grad_(False)
		for name, mod in list(model.named_modules()):
			for child_name, child in list(mod.named_children()):
				if child_name in targets and isinstance(child, nn.Linear):
					# replace child with our linear layer
					lora = LoRALinear(child, **kw).to(child.weight.device, child.weight.dtype)
					setattr(mod, child_name, lora)
		return model


	target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
	model = inject_lora(model, target_modules)
	opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)


	############### TRAIN ################

	for i, batch in enumerate(dl):
		batch = {k: v.cuda() for k, v in batch.items()}
		loss = model(**batch).loss / accum
		loss.backward()
		opt.step()
		opt.zero_grad()

	# save
	torch.save({k: v for k, v in model.state_dict().items() if ".A" in k or ".B" in k}, "lora.pt")

	# merge
	# for m in model.modules():
	# 	if isinstance(m, LoRALinear):
	# 		m.base.weight.data += m.scale * (m.B @ m.A)




@app.local_entrypoint()
def main(workers: int = 2):
	process_data.remote()
	# train.remote()