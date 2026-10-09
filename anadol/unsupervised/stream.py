"""
Live-stream an endless StyleGAN2 latent walk (WikiArt_Uncond2.pkl in Modal volume "anadol")
to Cloudflare Stream over RTMPS.

Setup (once):
  1. Cloudflare dashboard -> Stream -> Live inputs -> Create live input
     Copy the RTMPS URL and RTMPS key.
  2. Fill in .env (see .env.example), then:
       modal secret create cloudflare-stream --from-dotenv .env --force
Run:
  modal run --detach stream.py --seed 42
  modal run --detach stream.py --seed 42 --mode walk --fps 24 --size 720 --minutes 30
Stop:
  modal app stop anadol-stream

Options:
  --seed      random seed
  --mode      perlin | walk
  --fps       frames per second (default 30; lower it if logs say the GPU is behind)
  --speed     perlin: noise cycles per second (default 0.3) | walk: step size (default 0.3)
  --amp       perlin: drift distance from start (default 1.5)
  --psi       truncation (default 0.8)
  --size      output width/height in px, 0 = model resolution (default 0)
  --bitrate   video bitrate (default 6M; Cloudflare recommends <= ~8M)
  --minutes   stop after N minutes, 0 = run until stopped (max 24 h)
  --model     .pkl in the volume; partial names work (e.g. ffhq, cat, modern_art)
  --gpu       Modal GPU type (default L4; needs NVENC: L4, T4, A10G, L40S)
"""
import os

import modal

VOLUME_NAME = "anadol"
PKL_NAME = "WikiArt_Uncond2.pkl"
MOUNT = "/anadol"


def find_pkl(name: str) -> str:
    """Find a .pkl in the volume by exact name or unique partial match (case-insensitive)."""
    import glob
    import os
    files = sorted(glob.glob(f"{MOUNT}/**/*.pkl", recursive=True))
    names = [os.path.basename(f) for f in files]
    q = name.lower().removesuffix(".pkl")
    exact = [f for f, n in zip(files, names) if n.lower().removesuffix(".pkl") == q]
    hits = exact or [f for f, n in zip(files, names) if q in n.lower()]
    if len(hits) == 1:
        return hits[0]
    listing = "\n  ".join(names) or "(none)"
    why = "matches several models" if hits else "matches no model"
    raise FileNotFoundError(f"--model '{name}' {why}. Models in volume '{VOLUME_NAME}':\n  {listing}")

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
        "TORCH_EXTENSIONS_DIR": "/anadol/.torch_extensions",
        "NVIDIA_DRIVER_CAPABILITIES": "all",  # exposes NVENC to ffmpeg
    })
)

app = modal.App("anadol-stream", image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


# ---------------------------------------------------------------------------
# Endless latent paths. Each z is rescaled to norm sqrt(z_dim) to stay in distribution.
# ---------------------------------------------------------------------------
def _renorm(z):
    import numpy as np
    return z / np.linalg.norm(z, axis=-1, keepdims=True) * np.sqrt(z.shape[-1])


class PerlinZ:
    """Start z + per-dimension 1D Perlin noise. Gradients come from a hash of
    (lattice index, dim), so the path is infinite and deterministic per seed."""

    def __init__(self, seed, z_dim, speed, amp, octaves=2):
        import numpy as np
        rng = np.random.RandomState(seed)
        self.base = rng.randn(z_dim)
        self.dims = np.arange(z_dim, dtype=np.uint64)[None, :]
        self.octaves = [
            (speed * 2 ** o, 0.5 ** o, rng.rand(1, z_dim) * 100, np.uint64(rng.randint(1, 2**31)))
            for o in range(octaves)
        ]
        self.scale = amp * 2 * np.sqrt(3)  # uniform[-1,1] gradients -> match unit-variance scale

    def _grad(self, i, salt):
        import numpy as np
        with np.errstate(over="ignore"):
            x = (i.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) \
                ^ (self.dims * np.uint64(0xBF58476D1CE4E5B9)) ^ (salt * np.uint64(0x94D049BB133111EB))
            x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
            x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
            x ^= x >> np.uint64(31)
        return (x >> np.uint64(11)).astype(np.float64) / 2.0**53 * 2 - 1

    def __call__(self, t):
        import numpy as np
        t = np.asarray(t, dtype=np.float64)[:, None]
        offset = 0.0
        for freq, weight, phase, salt in self.octaves:
            x = t * freq + phase
            i0 = np.floor(x)
            f = x - i0
            g0, g1 = self._grad(i0, salt), self._grad(i0 + 1, salt)
            fade = f * f * f * (f * (f * 6 - 15) + 10)
            offset = offset + weight * ((1 - fade) * g0 * f + fade * g1 * (f - 1))
        return _renorm(self.base + self.scale * offset)


class WalkZ:
    """Smooth random walk: velocity with momentum so direction changes gradually."""

    def __init__(self, seed, z_dim, speed, momentum=0.95):
        import numpy as np
        self.rng = np.random.RandomState(seed)
        self.z, self.v = self.rng.randn(z_dim), np.zeros(z_dim)
        self.step, self.momentum = speed * 0.035, momentum

    def __call__(self, t):
        import numpy as np
        out = np.empty((len(t), len(self.z)))
        for k in range(len(t)):
            self.v = self.momentum * self.v + self.step * self.rng.randn(len(self.z))
            self.z = _renorm(self.z + self.v)
            out[k] = self.z
        return out


# ---------------------------------------------------------------------------
def _pick_encoder():
    import subprocess
    ok = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=size=256x256:rate=30",
         "-frames:v", "5", "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True).returncode == 0
    if not ok:
        print("NVENC unavailable -> using libx264 on CPU")
    return "h264_nvenc" if ok else "libx264"


def _ffmpeg_cmd(res, size, fps, bitrate, encoder, dst):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-stats_period", "10",
           "-thread_queue_size", "256",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{res}x{res}", "-r", str(fps), "-i", "-",
           "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",  # silent audio track
           "-map", "0:v", "-map", "1:a", "-shortest"]
    if size and size != res:
        cmd += ["-vf", f"scale={size}:{size}:flags=lanczos"]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "cbr", "-bf", "0",
                "-forced-idr", "1", "-no-scenecut", "1"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"]
    cmd += ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
            "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-keyint_min", str(fps * 2), "-sc_threshold", "0",
            "-force_key_frames", "expr:gte(t,n_forced*2)",  # IDR keyframe exactly every 2 s
            "-c:a", "aac", "-b:a", "128k", "-f", "flv", dst]
    return cmd


@app.cls(
    gpu="L4",
    volumes={MOUNT: vol},
    secrets=[modal.Secret.from_name("anadol-stream")],
    timeout=24 * 60 * 60,
)
class Streamer:
    model: str = modal.parameter(default=PKL_NAME)

    @modal.enter()
    def load(self):
        import glob
        import torch
        import legacy

        path = find_pkl(self.model)
        with open(path, "rb") as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].cuda().eval()
        print(f"Loaded {path}: z_dim={self.G.z_dim} res={self.G.img_resolution}")
        with torch.inference_mode():  # warm up / compile CUDA ops (cached in the volume)
            self.G(torch.zeros(1, self.G.z_dim, device="cuda"),
                   torch.zeros(1, self.G.c_dim, device="cuda"), noise_mode="const")
        vol.commit()

    @modal.method()
    def stream(self, seed: int, mode: str, fps: int, speed: float, amp: float, psi: float,
               size: int, bitrate: str, minutes: float):
        import queue
        import subprocess
        import threading
        import time

        import numpy as np
        import torch

        url = os.environ["CF_RTMPS_URL"].rstrip("/")
        key = os.environ.get("CF_STREAM_KEY", "")
        dst = f"{url}/{key}" if key else url
        if os.environ.get("CF_CUSTOMER_CODE") and os.environ.get("CF_LIVE_INPUT_ID"):
            print("Watch: https://customer-{}.cloudflarestream.com/{}/iframe".format(
                os.environ["CF_CUSTOMER_CODE"], os.environ["CF_LIVE_INPUT_ID"]))

        G = self.G
        res = G.img_resolution
        path = PerlinZ(seed, G.z_dim, speed, amp) if mode == "perlin" else WalkZ(seed, G.z_dim, speed)
        encoder = _pick_encoder()
        end_time = time.time() + minutes * 60 if minutes > 0 else float("inf")

        # Producer thread: GPU renders frames in batches into a small queue
        frames = queue.Queue(maxsize=fps * 2)
        stop = threading.Event()
        producer_error = []

        def produce():
            try:
                batch, i, t_start = 8, 0, time.time()
                c = torch.zeros(batch, G.c_dim, device="cuda")
                with torch.inference_mode():
                    while not stop.is_set():
                        t = (np.arange(batch) + i) / fps
                        z = torch.from_numpy(path(t)).float().cuda()
                        img = G(z, c, truncation_psi=psi, noise_mode="const")
                        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        if i == 0:
                            print(f"First batch rendered: {img.shape} in {time.time() - t_start:.2f}s")
                        for f in img:
                            while not stop.is_set():
                                try:
                                    frames.put(f.tobytes(), timeout=0.5)
                                    break
                                except queue.Full:
                                    pass
                        i += batch
                        if i % (fps * 30) < batch:
                            print(f"GPU render speed: {i / (time.time() - t_start):.1f} fps (target {fps})")
            except Exception as e:
                import traceback
                traceback.print_exc()
                producer_error.append(e)
                stop.set()

        threading.Thread(target=produce, daemon=True).start()

        # Writer: pace frames to real time and push to Cloudflare; reconnect on failure
        sent = 0
        try:
            while time.time() < end_time:
                print(f"Connecting to Cloudflare ({encoder}, {fps} fps, {size or res}px, {bitrate})...")
                enc = subprocess.Popen(_ffmpeg_cmd(res, size, fps, bitrate, encoder, dst),
                                       stdin=subprocess.PIPE)
                t0, n, last_log = time.time(), 0, time.time()
                try:
                    while time.time() < end_time:
                        try:
                            frame = frames.get(timeout=30)
                        except queue.Empty:
                            if producer_error:
                                raise RuntimeError("Frame renderer crashed (traceback above)") from producer_error[0]
                            print("WARNING: no frames rendered for 30 s")
                            continue
                        if sent == 0:
                            print("First frame sent to Cloudflare")
                        delay = t0 + n / fps - time.time()
                        if delay > 0:
                            time.sleep(delay)
                        elif delay < -2:  # fell behind: reset the clock instead of bursting
                            print(f"WARNING: GPU is {-delay:.1f}s behind real time; try a lower --fps or --size")
                            t0, n = time.time(), 0
                        enc.stdin.write(frame)
                        n += 1
                        sent += 1
                        if time.time() - last_log > 30:
                            print(f"streaming: {sent} frames sent, queue={frames.qsize()}")
                            last_log = time.time()
                    break  # reached --minutes
                except (BrokenPipeError, OSError):
                    print("Connection lost; reconnecting in 3 s...")
                    time.sleep(3)
                finally:
                    try:
                        enc.stdin.close()
                        enc.wait(timeout=10)
                    except Exception:
                        enc.kill()
        finally:
            stop.set()
        print(f"Stream ended after {sent} frames.")


@app.local_entrypoint()
def main(seed: int = 0, mode: str = "perlin", fps: int = 30, speed: float = 0.1, amp: float = 1.5,
         psi: float = 0.6, size: int = 0, bitrate: str = "6M", minutes: float = 0, gpu: str = "L4",
         model: str = PKL_NAME):
    if mode not in ("perlin", "walk"):
        raise SystemExit("--mode must be 'perlin' or 'walk'")
    streamer = Streamer.with_options(gpu=gpu)(model=model)
    streamer.stream.remote(seed, mode, fps, speed, amp, psi, size, bitrate, minutes)