"""
Render a short latent-walk video from WikiArt_Uncond2.pkl (StyleGAN2-ADA) in the Modal volume "anadol".

Usage:
  modal run video.py --seed 42                              # 5 s Perlin-noise walk
  modal run video.py --seed 42 --mode walk                  # smooth random walk (momentum)
  modal run video.py --seed 7 --seconds 10 --fps 30 --speed 0.5 --psi 0.7

Options:
  --seed      random seed (same seed = same video)
  --mode      perlin | walk
  --seconds   video length (default 5)
  --fps       frames per second (default 30)
  --speed     perlin: noise cycles per second (default 0.3)
              walk:   step size per frame (default 0.3 -> scaled internally)
  --amp       perlin: how far z drifts from its start (default 1.5)
  --psi       truncation (lower = cleaner, less varied; default 0.8)
Output: walk_s<seed>_<mode>.mp4 in your current folder.
"""
import modal

VOLUME_NAME = "anadol"
PKL_NAME = "WikiArt_Uncond2.pkl"
MOUNT = "/anadol"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "ffmpeg")
    .pip_install("torch", "numpy", "pillow", "click", "requests", "scipy", "ninja",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
    .run_commands(
        "git clone https://github.com/NVlabs/stylegan2-ada-pytorch /stylegan2-ada-pytorch",
        # Patch for newer PyTorch: use the module returned by cpp_extension.load()
        """python -c "p='/stylegan2-ada-pytorch/torch_utils/custom_ops.py'; s=open(p).read(); """
        """s=s.replace('torch.utils.cpp_extension.load(name=module_name', 'module = torch.utils.cpp_extension.load(name=module_name'); """
        """s=s.replace('module = importlib.import_module(module_name)', 'pass'); open(p,'w').write(s)" """,
    )
    .env({
        "PYTHONPATH": "/stylegan2-ada-pytorch",
        "TORCH_EXTENSIONS_DIR": "/anadol/.torch_extensions",  # compiled ops cached in the volume
    })
)

app = modal.App("anadol-video", image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


# ---------------------------------------------------------------------------
# Latent paths. Each returns an array of shape (n_frames, z_dim).
# Every z is rescaled to norm sqrt(z_dim) so it stays "in distribution".
# ---------------------------------------------------------------------------
def _renorm(z):
    import numpy as np
    return z / np.linalg.norm(z, axis=-1, keepdims=True) * np.sqrt(z.shape[-1])


def perlin_path(seed, n_frames, z_dim, fps, speed, amp, octaves=2):
    """Start z + per-dimension 1D Perlin noise (random phase per dim so they don't sync)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    base = rng.randn(z_dim)
    t = np.arange(n_frames)[:, None] / fps                       # (F, 1) seconds
    offset = np.zeros((n_frames, z_dim))
    for o in range(octaves):
        freq, weight = speed * 2 ** o, 0.5 ** o
        x = t * freq + rng.rand(1, z_dim) * 100                  # (F, D) lattice coords
        i0 = np.floor(x).astype(np.int64)
        f = x - i0
        n_pts = int(i0.max() - i0.min()) + 2
        grads = rng.randn(n_pts, z_dim)                          # gradient per lattice point per dim
        j = i0 - i0.min()
        cols = np.arange(z_dim)[None, :]
        g0, g1 = grads[j, cols], grads[j + 1, cols]
        fade = f * f * f * (f * (f * 6 - 15) + 10)
        offset += weight * ((1 - fade) * g0 * f + fade * g1 * (f - 1))
    return _renorm(base + amp * 2 * offset)                     # perlin ~[-0.5,0.5] -> scale up


def walk_path(seed, n_frames, z_dim, speed, momentum=0.95):
    """Smooth random walk: velocity with momentum, so direction changes gradually."""
    import numpy as np
    rng = np.random.RandomState(seed)
    z, v = rng.randn(z_dim), np.zeros(z_dim)
    step = speed * 0.035
    out = np.empty((n_frames, z_dim))
    for i in range(n_frames):
        v = momentum * v + step * rng.randn(z_dim)
        z = _renorm(z + v)
        out[i] = z
    return out


@app.cls(gpu="L4", volumes={MOUNT: vol}, scaledown_window=300, timeout=1800)
class Renderer:
    @modal.enter()
    def load(self):
        import glob
        import torch
        import legacy  # stylegan2-ada-pytorch (converts TF pickles automatically)

        matches = glob.glob(f"{MOUNT}/**/{PKL_NAME}", recursive=True)
        if not matches:
            raise FileNotFoundError(f"{PKL_NAME} not found in volume '{VOLUME_NAME}'")
        with open(matches[0], "rb") as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].cuda().eval()
        print(f"Loaded {matches[0]}: z_dim={self.G.z_dim} res={self.G.img_resolution}")
        with torch.inference_mode():  # warm up / compile CUDA ops, cache them in the volume
            self.G(torch.zeros(1, self.G.z_dim, device="cuda"),
                   torch.zeros(1, self.G.c_dim, device="cuda"), noise_mode="const")
        vol.commit()

    @modal.method()
    def render(self, seed: int, mode: str, seconds: float, fps: int,
               speed: float, amp: float, psi: float) -> bytes:
        import subprocess
        import tempfile
        import time
        import torch

        G = self.G
        n = max(1, round(seconds * fps))
        if mode == "perlin":
            zs = perlin_path(seed, n, G.z_dim, fps, speed, amp)
        elif mode == "walk":
            zs = walk_path(seed, n, G.z_dim, speed)
        else:
            raise ValueError("mode must be 'perlin' or 'walk'")

        res = G.img_resolution
        out_path = tempfile.mktemp(suffix=".mp4")
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{res}x{res}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", out_path],
            stdin=subprocess.PIPE,
        )

        t0, batch = time.time(), 8
        c = torch.zeros(batch, G.c_dim, device="cuda")
        with torch.inference_mode():
            for i in range(0, n, batch):
                z = torch.from_numpy(zs[i:i + batch]).float().cuda()
                img = G(z, c[: len(z)], truncation_psi=psi, noise_mode="const")  # (B,3,H,W) [-1,1]
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                enc.stdin.write(img.cpu().numpy().tobytes())
        enc.stdin.close()
        enc.wait()
        print(f"Rendered {n} frames at {res}x{res} in {time.time() - t0:.1f}s")
        with open(out_path, "rb") as f:
            return f.read()


@app.local_entrypoint()
def main(seed: int = 0, mode: str = "perlin", seconds: float = 5.0, fps: int = 30,
         speed: float = 0.3, amp: float = 1.5, psi: float = 0.8):
    mp4 = Renderer().render.remote(seed, mode, seconds, fps, speed, amp, psi)
    out = f"walk_s{seed}_{mode}.mp4"
    with open(out, "wb") as f:
        f.write(mp4)
    print("Saved", out)