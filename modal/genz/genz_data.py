

# This files is for these
# https://www.tensorflow.org/datasets/catalog/reddit # 2GB
# https://www.tensorflow.org/datasets/catalog/reddit_tifu 600 MB
# tiagozip/tiktok # masssivve 208 GB


import os
import tiktoken
import urllib.request
import modal
import json

app = modal.App("genz-data")

tfimage = (
	modal.Image
		.debian_slim()
		.pip_install(
			"tensorflow", "tensorflow-datasets", 
			"tiktoken", "numpy", "importlib_resources",
			"bs4")
)
hfimage = (
	modal.Image
		.debian_slim()
		.pip_install("datasets", "tiktoken", "numpy")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("genz", create_if_missing=True)

def get_last_shard_ix():
	import os
	import re
	pat = re.compile(r"^bigdata_(\d+)\.bin$")
	existing_shards = [int(m.group(1)) for p in os.listdir("/data/big") if (m := pat.match(p))]
	last_shard_idx = max(existing_shards, default=-1)
	return last_shard_idx


@app.function(image=hfimage, cpu=4, volumes={"/data": vol}, timeout=60 * 60)
def process_tiktok_comments(args):
	i, hf_path = args

	import os
	import re
	import numpy as np
	from datasets import load_dataset
	print("starting hf_path: ", hf_path)

	m = re.search(r"(\d{4})-(\d{2})\.parquet$", hf_path)
	year, month = int(m.group(1)), int(m.group(2))
	ym = f"{year}{month:02d}" 

	comments = load_dataset("tiagozip/tiktok", data_files=hf_path, split="train", streaming=True)

	os.makedirs("/data/big", exist_ok=True)

	enc = tiktoken.get_encoding('gpt2')
	EOT = enc.eot_token

	SHARD = 10_000_000
	ids = [EOT]
	k = 0
	for row in comments:
		if not row["text"]:
			continue

		curids = enc.encode_ordinary(row["text"]) + [EOT]
		ids.extend(curids)

		if len(ids) > SHARD:
			np.array(ids, dtype=np.uint16).tofile(f"/data/big/tiktok_{ym}_{k:03d}.bin")
			k += 1
			ids = [EOT]
	
	if len(ids) <= 1:
		np.array(ids, dtype=np.uint16).tofile(f"/data/big/tiktok_{ym}_{k:03d}.bin")

	with open("/data/tiktok_done.txt", "a") as f:
		f.write(hf_path + "\n")

	vol.commit()

	print(f"succcess, finished tiktok task {hf_path}")
	return 1





@app.function(image=tfimage, cpu=4, volumes={"/data": vol}, timeout=60 * 60 * 6)
def process_reddit():
	# 2 GB
	import os
	import numpy as np
	import tensorflow_datasets as tfds
	os.makedirs("/data/big", exist_ok=True)

	last_ix = get_last_shard_ix()
	enc = tiktoken.get_encoding('gpt2')
	EOT = enc.eot_token

	SHARD = 10_000_000
	ids = [EOT]
	ds = tfds.load("reddit", split="train")
	for ex in tfds.as_numpy(ds):
		content = enc.encode_ordinary(ex["content"].decode())
		summary = enc.encode_ordinary(ex["summary"].decode())
		ids.extend(content + summary + [EOT])

		if len(ids) > SHARD:
			# flush
			last_ix += 1
			np.array(ids, dtype=np.uint16).tofile(f"/data/big/reddit_{last_ix}.bin")
			ids = [EOT]
	
	if len(ids) <= 1:
		last_ix += 1
		np.array(ids, dtype=np.uint16).tofile(f"/data/big/reddit_{last_ix}.bin")

	vol.commit()
	print("succcess, finished reddit task")
	return 1

@app.function(image=tfimage, cpu=4, volumes={"/data": vol}, timeout=60 * 60)
def process_reddit_tifu():
	# 639.54 MB
	import os
	import numpy as np
	import tensorflow_datasets as tfds
	os.makedirs("/data/big", exist_ok=True)

	last_ix = get_last_shard_ix()
	enc = tiktoken.get_encoding('gpt2')
	EOT = enc.eot_token

	SHARD = 10_000_000
	ids = [EOT]
	ds = tfds.load("reddit_tifu/long", split="train")
	for ex in tfds.as_numpy(ds):
		post = enc.encode_ordinary(ex["documents"].decode())
		tldr = enc.encode_ordinary(ex["tldr"].decode())
		ids.extend(post + tldr + [EOT])

		if len(ids) > SHARD:
			# flush
			last_ix += 1
			np.array(ids, dtype=np.uint16).tofile(f"/data/big/bigdata_{last_ix}.bin")
			ids = [EOT]
			vol.commit()
	
	if len(ids) <= 1:
		last_ix += 1
		np.array(ids, dtype=np.uint16).tofile(f"/data/big/bigdata_{last_ix}.bin")

	vol.commit()

	print("succcess, finished reddit tifu task")
	return 1

@app.local_entrypoint()
def main():
	# process_reddit_tifu.remote()
	process_reddit.remote()

	# from huggingface_hub import list_repo_files
	# files = [f for f in list_repo_files("tiagozip/tiktok", repo_type="dataset")
	# 	if f.endswith(".parquet") and "comments" in f]
		
	# # map this
	# process_tiktok_comments.map(enumerate(files))
	# print("success")


