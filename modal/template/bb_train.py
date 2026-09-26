import modal

app = modal.App("mnist-train")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "torchvision")
		.add_local_python_source("train_mnist")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("mnist-vol", create_if_missing=True)


@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=60 * 30)
def train(epochs: int = 3, batch_size: int = 128, lr: float = 1e-3):
	from train_mnist import main
	acc = main(epochs=epochs, out_dir="/data")
	vol.commit()
	return acc

@app.local_entrypoint()
def main(epochs: int = 3):
	print("final acc:", train.remote(epochs))