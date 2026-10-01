

# This files is for these
# https://www.tensorflow.org/datasets/catalog/reddit # 2GB
# https://www.tensorflow.org/datasets/catalog/reddit_tifu 600 MB
# tiagozip/tiktok # masssivve 208 GB


import os
import torch
import torch.nn as nn
import tiktoken
import urllib.request
import modal
import json

app = modal.App("genz-data")

image = (
	modal.Image
		.debian_slim()
		.pip_install("datasets", "tiktoken")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("genz", create_if_missing=True)

context_size = 256
batchsize = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# @app.function(image=image, gpu="T4", volumes={"/ckpts": vol}, timeout=60 * 60)
def process_aiml8726737():
	from datasets import load_dataset
	ds = load_dataset("aiml8726737/gen-alpha-brainrot", num_proc=8)
	print(ds["messages"])
	return ds

# @app.function(image=image, gpu="T4", volumes={"/ckpts": vol}, timeout=60 * 60)
def process_AliceYin():
	from datasets import load_dataset
	ds = load_dataset("AliceYin/genz-contextual-abusive-slang-v2-silver", num_proc=8)
	enc = tiktoken.get_encoding('gpt2')

	return data, val_data, enc

@app.local_entrypoint()
def main():
	process_AliceYin.remote()
	process_aiml8726737.remote()

if __name__ == "__main__":
	process_aiml8726737()


