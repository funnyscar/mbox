# works, but funky. prefer individual files the represent newest upgrades to the model

import modal

app = modal.App("minigpt")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken")
		.add_local_python_source("minigpt")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("minigpt", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/ckpts": vol}, timeout=60 * 60)
def train():
	from minigpt import train
	loss = train(vol, epochs=10000, out_path="/ckpts/minigpt.pt")
	vol.commit()
	return loss

@app.local_entrypoint()
def main():
	print("final loss:", train.remote())
