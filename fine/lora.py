import os
import torch
import torch.nn as nn
import tiktoken
import modal
import json
import math

app = modal.App("fine-lora")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "datasets", "transformers", "peft", "trl", "bitsandbytes")
		.add_local_python_source("model")
)
dataimage = (
	modal.Image
		.debian_slim()
		.pip_install("datasets")
)

vol = modal.Volume.from_name("fine", create_if_missing=True)


@app.function(image=dataimage, cpu=4, volumes={"/data": vol}, timeout=60 * 60)
def process_data(state_dict, total_steps=5000):
	ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(10_000))
	ds.select_columns(["messages"]).to_json("/data/train.jsonl")
	print("done")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def train(state_dict, total_steps=5000):
	import torch
	from datasets import load_dataset
	from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
	from peft import LoraConfig
	from trl import SFTTrainer, SFTConfig

	model_id = "Qwen/Qwen3-8B"  # or a smaller Qwen for testing

	# W·x + (alpha/r) · B·A·x

	tok = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=BitsAndBytesConfig(
			load_in_4bit=True, bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.bfloat16),
		device_map="auto",
	)

	# JSONL with {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}
	ds = load_dataset("json", data_files="/data/train.jsonl", split="train")

	lora = LoraConfig(
		r=16, lora_alpha=32, lora_dropout=0.05,
		target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
		task_type="CAUSAL_LM",
	)

	trainer = SFTTrainer(
		model=model, train_dataset=ds, peft_config=lora, processing_class=tok,
		args=SFTConfig(
			output_dir="qwen-lora", num_train_epochs=2,
			per_device_train_batch_size=2, gradient_accumulation_steps=8,
			learning_rate=2e-4, bf16=True, logging_steps=10,
			gradient_checkpointing=True, max_length=2048,
		),
	)
	trainer.train()

	# save adapter. LoraConfig + safetensors
	trainer.save_model("qwen-lora")

	# merge 
	from peft import PeftModel
	base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
	merged = PeftModel.from_pretrained(base, "qwen-lora").merge_and_unload()
	merged.save_pretrained("qwen-merged")



@app.local_entrypoint()
def main(workers: int = 2):
	data
	pass