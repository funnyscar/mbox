"""
Generate one image from WikiArt_Uncond2.pkl (StyleGAN2-ADA) stored in the Modal volume "anadol".

Usage:
  modal run generate.py --seed 42
  modal run generate.py --seed 42 --psi 0.7          # truncation (lower = safer/less varied)
Output: seed0042.png in your current folder.
"""
import modal

VOLUME_NAME = "anadol"
PKL_NAME = "WikiArt_Uncond2.pkl"
MOUNT = "/anadol"

image = (
    # CUDA *devel* image: includes nvcc so StyleGAN's custom CUDA ops (upfirdn2d, bias_act) compile
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch", "numpy", "pillow", "click", "requests", "scipy", "ninja",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
    .run_commands(
        "git clone https://github.com/NVlabs/stylegan2-ada-pytorch /stylegan2-ada-pytorch",
        # Patch for newer PyTorch: use the module returned by cpp_extension.load()
        # instead of importlib.import_module(), which no longer finds it.
        """python -c "p='/stylegan2-ada-pytorch/torch_utils/custom_ops.py'; s=open(p).read(); """
        """s=s.replace('torch.utils.cpp_extension.load(name=module_name', 'module = torch.utils.cpp_extension.load(name=module_name'); """
        """s=s.replace('module = importlib.import_module(module_name)', 'pass'); open(p,'w').write(s)" """,
    )
    .env({
        "PYTHONPATH": "/stylegan2-ada-pytorch",
        # cache compiled ops in the volume so they only build once (~1 min), not every cold start
        "TORCH_EXTENSIONS_DIR": "/anadol/.torch_extensions",
    })
)

app = modal.App("anadol-generate", image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


@app.cls(gpu="L4", volumes={MOUNT: vol}, scaledown_window=300)
class Generator:
    @modal.enter()
    def load(self):
        import glob
        import torch
        import legacy  # from stylegan2-ada-pytorch (converts TF pickles automatically)

        matches = glob.glob(f"{MOUNT}/**/{PKL_NAME}", recursive=True)
        if not matches:
            raise FileNotFoundError(f"{PKL_NAME} not found in volume '{VOLUME_NAME}'")
        print("Loading", matches[0])
        with open(matches[0], "rb") as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].cuda().eval()
        print(f"Loaded: z_dim={self.G.z_dim} c_dim={self.G.c_dim} res={self.G.img_resolution}")

        # Warm up: triggers the CUDA op compile now, then save the build to the volume
        with torch.inference_mode():
            self.G(torch.zeros(1, self.G.z_dim, device="cuda"),
                   torch.zeros(1, self.G.c_dim, device="cuda"), noise_mode="const")
        vol.commit()

    @modal.method()
    def generate(self, seed: int, psi: float = 1.0) -> bytes:
        import io
        import numpy as np
        import torch
        from PIL import Image

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).cuda()
        c = torch.zeros(1, self.G.c_dim, device="cuda")  # empty for unconditional
        with torch.inference_mode():
            img = self.G(z, c, truncation_psi=psi, noise_mode="const")  # (1,3,H,W) in [-1,1]
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        buf = io.BytesIO()
        Image.fromarray(img[0].cpu().numpy(), "RGB").save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def main(seed: int = 0, psi: float = 1.0):
    png = Generator().generate.remote(seed, psi)
    out = f"seed{seed:04d}.png"
    with open(out, "wb") as f:
        f.write(png)
    print("Saved", out)