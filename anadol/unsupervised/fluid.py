"""
StyleGAN2 Perlin walk in W space (not z), viewed straight off the GPU. No Cloudflare.
Same API/flow as gpu_view.py; deploys separately as "gpu-view-w".

Why W: the mapping network turns z into w, the style vector the synthesis network actually
uses. Walking w directly gives smoother, more "semantic" motion, and you can restrict the
motion to some synthesis layers (coarse = pose/layout, middle = features, fine = color/texture).

Setup (once, shared with gpu_view.py):
  modal secret create gpu-view API_KEY=$(openssl rand -hex 16)
Run one stream from the terminal:
  modal run gpu_view_w.py --model ffhq --seed 42
  modal run gpu_view_w.py --model ffhq --space wplus --layers 0-3      # only move coarse layers
  modal run gpu_view_w.py --model modern_art --layers 8-15 --amp 1.5   # only move fine layers
  modal run gpu_view_w.py --model modern_art --fluid 4 --fluid-mix 0.06 # strong fluid, long trails
  modal run gpu_view_w.py --model ffhq --fluid 0                       # no fluid
Or deploy the API (same endpoints as gpu_view.py):
  modal deploy gpu_view_w.py
  curl -X POST "$API/streams" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
       -d '{"model":"ffhq","seed":42,"space":"wplus","layers":"0-7"}'

Options (CLI flags / POST fields):
  model         .pkl in the "anadol" volume; partial names work     (default WikiArt_Uncond2)
  seed          starting w = mapping(z from this seed)              (default 0)
  speed         Perlin cycles per second                             (default 0.15)
  amp           drift size in units of W's per-dimension std         (default 1.0)
  space         "w": one moving w shared by the chosen layers;
                "wplus": each chosen layer drifts independently      (default w)
  layers        synthesis layers that move, e.g. "all", "0-3", "0,1,8-11"   (default all)
  psi           truncation toward the average w                      (default 0.8)
  fluid         Anadol-style fluid motion, pixels/frame at 512 px; 0 = off   (default 2.0)
  fluid_mix     fresh GAN pixels mixed in per frame; lower = longer trails   (default 0.12)
  fluid_scale   swirl size: cells across the frame; lower = bigger swirls    (default 4)
  fluid_follow  0 = pure swirl, 1 = flow along the image's contours          (default 0.5)
  fluid_speed   how fast the swirl pattern itself changes                    (default 0.5)
  fps, size, quality, idle_minutes, max_minutes, gpu                  (as in gpu_view.py)
"""
import os
import time

import modal

VOLUME_NAME = "anadol"
PKL_NAME = "textures.pkl"
MOUNT = "/anadol"
PORT = 8000
MAX_STREAMS = 3                      # max simultaneous GPU containers
GPUS = {"L4", "A10G", "L40S", "T4"}

app = modal.App("gpu-view-w")
vol = modal.Volume.from_name(VOLUME_NAME)
sessions = modal.Dict.from_name("gpu-view-w-sessions", create_if_missing=True)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch", "torchvision", "numpy", "pillow", "click", "requests", "scipy", "ninja", "aiohttp",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
    .run_commands(
        "git clone https://github.com/NVlabs/stylegan2-ada-pytorch /stylegan2-ada-pytorch",
        # Patch for newer PyTorch: use the module returned by cpp_extension.load()
        """python -c "p='/stylegan2-ada-pytorch/torch_utils/custom_ops.py'; s=open(p).read(); """
        """s=s.replace('torch.utils.cpp_extension.load(name=module_name', 'module = torch.utils.cpp_extension.load(name=module_name'); """
        """s=s.replace('module = importlib.import_module(module_name)', 'pass'); open(p,'w').write(s)" """,
    )
    .env({"PYTHONPATH": "/stylegan2-ada-pytorch", "TORCH_EXTENSIONS_DIR": "/anadol/.torch_extensions"})
)
api_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")


def list_models():
    import glob
    return sorted(os.path.basename(f) for f in glob.glob(f"{MOUNT}/**/*.pkl", recursive=True))


def find_pkl(name: str) -> str:
    """Exact name or unique partial match (case-insensitive) -> basename. Raises ValueError otherwise."""
    names = list_models()
    q = name.lower().removesuffix(".pkl")
    exact = [n for n in names if n.lower().removesuffix(".pkl") == q]
    hits = exact or [n for n in names if q in n.lower()]
    if len(hits) == 1:
        return hits[0]
    why = "matches several models" if hits else "matches no model"
    raise ValueError(f"model '{name}' {why}. Available: {', '.join(names) or '(none)'}")


# ---------------------------------------------------------------------------
# Perlin noise over many dimensions, normalised to ~unit std per dimension.
# Gradients come from a hash of (lattice index, dim): infinite, smooth, deterministic per seed.
# ---------------------------------------------------------------------------
class PerlinNoise:
    def __init__(self, seed, dims, speed, octaves=2):
        import numpy as np
        rng = np.random.RandomState(seed + 1_000_003)
        self.dims = np.arange(dims, dtype=np.uint64)[None, :]
        self.octaves = [(speed * 2 ** o, 0.5 ** o, rng.rand(1, dims) * 100, np.uint64(rng.randint(1, 2**31)))
                        for o in range(octaves)]
        probe = self._raw(np.linspace(0, 200 / max(speed, 1e-6), 400))
        self.norm = 1.0 / max(float(probe.std()), 1e-8)

    def _grad(self, i, salt):
        import numpy as np
        with np.errstate(over="ignore"):
            x = (i.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) \
                ^ (self.dims * np.uint64(0xBF58476D1CE4E5B9)) ^ (salt * np.uint64(0x94D049BB133111EB))
            x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
            x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
            x ^= x >> np.uint64(31)
        return (x >> np.uint64(11)).astype(np.float64) / 2.0**53 * 2 - 1

    def _raw(self, t):
        import numpy as np
        t = np.asarray(t, dtype=np.float64)[:, None]
        out = 0.0
        for freq, weight, phase, salt in self.octaves:
            x = t * freq + phase
            i0 = np.floor(x)
            f = x - i0
            g0, g1 = self._grad(i0, salt), self._grad(i0 + 1, salt)
            fade = f * f * f * (f * (f * 6 - 15) + 10)
            out = out + weight * ((1 - fade) * g0 * f + fade * g1 * (f - 1))
        return out

    def __call__(self, t):
        """(len(t), dims) offsets with ~unit std per dimension."""
        return self._raw(t) * self.norm


def parse_layers(spec, num_ws):
    """'all' | '0-3' | '0,1,8-11' -> sorted list of layer indices, validated against num_ws."""
    spec = str(spec).strip().lower()
    if spec in ("", "all"):
        return list(range(num_ws))
    out = set()
    try:
        for part in spec.replace(" ", "").split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            elif part:
                out.add(int(part))
    except ValueError:
        raise ValueError(f"layers '{spec}' invalid: use e.g. all, 0-3, 0,1,8-11")
    bad = sorted(i for i in out if not 0 <= i < num_ws)
    if bad or not out:
        raise ValueError(f"layers '{spec}' invalid: this model has layers 0-{num_ws - 1}")
    return sorted(out)


# ---------------------------------------------------------------------------
# Fluid motion (Refik Anadol-style "data painting"): a feedback canvas that is advected
# every frame by a swirling, divergence-free velocity field, with fresh GAN pixels mixed in.
# All on the GPU with torch: curl noise + optional flow along image contours + grid_sample.
# ---------------------------------------------------------------------------
class FluidFX:
    def __init__(self, H, W, seed, strength=2.0, mix=0.12, scale=4.0, follow=0.5, flow_speed=0.5,
                 inertia=0.85, sharpen=0.35, device="cuda"):
        import torch
        self.torch = torch
        self.H, self.W, self.device = H, W, device
        self.strength, self.mix, self.follow, self.inertia = strength, mix, follow, inertia
        self.sharpen = sharpen                                     # counters blur from repeated resampling
        self.res = max(2, int(round(scale)))                      # noise cells across the frame
        self.period = 1.0 / max(flow_speed, 1e-3)                 # seconds between noise keyframes
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.keys = [self._key(), self._key()]
        self.key_t = 0.0
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.base = torch.stack((gx, gy), -1)[None]                # (1,H,W,2) sampling grid
        self.px = torch.tensor([2.0 / (W - 1), 2.0 / (H - 1)], device=device)
        self.canvas = None
        self.vel = None

    def _key(self):
        return self.torch.randn(1, 1, self.res + 3, self.res + 3, generator=self.gen, device=self.device)

    def _potential(self, t):
        F = self.torch.nn.functional
        while t - self.key_t >= self.period:                      # roll noise keyframes forward
            self.keys = [self.keys[1], self._key()]
            self.key_t += self.period
        u = (t - self.key_t) / self.period
        u = u * u * (3 - 2 * u)                                    # smoothstep between keyframes
        n = self.keys[0] * (1 - u) + self.keys[1] * u
        return F.interpolate(n, size=(self.H, self.W), mode="bicubic", align_corners=True)[0, 0]

    def __call__(self, frame, t):
        """frame: (3,H,W) float in [0,1] on the GPU -> fluid canvas, same shape."""
        torch, F = self.torch, self.torch.nn.functional
        psi = self._potential(t)
        dy, dx = torch.gradient(psi)
        v = torch.stack((dy, -dx))                                 # curl of psi: swirls, no sources/sinks
        v = v / (v.std() + 1e-8)
        if self.follow > 0:                                        # also flow along the image's contours
            lum = (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2])[None, None]
            lum = F.avg_pool2d(lum, 9, stride=1, padding=4)[0, 0]
            gy, gx = torch.gradient(lum)
            along = torch.stack((-gy, gx))
            along = along / (along.std() + 1e-8)
            v = v * (1 - self.follow) + along * self.follow
            v = v / (v.std() + 1e-8)
        v = v * self.strength                                      # pixels per frame
        self.vel = v if self.vel is None else self.inertia * self.vel + (1 - self.inertia) * v
        if self.canvas is None:
            self.canvas = frame.clone()
        grid = self.base - self.vel.permute(1, 2, 0)[None] * self.px   # semi-Lagrangian: look upstream
        adv = F.grid_sample(self.canvas[None], grid, mode="bilinear", padding_mode="reflection",
                            align_corners=True)[0]
        if self.sharpen > 0:                                       # keep the streaks crisp
            blur = F.avg_pool2d(adv[None], 3, stride=1, padding=1, count_include_pad=False)[0]
            adv = (adv + self.sharpen * (adv - blur)).clamp(0, 1)
        self.canvas = adv * (1 - self.mix) + frame * self.mix
        return self.canvas


# ---------------------------------------------------------------------------
# GPU side: render -> JPEG on GPU -> MJPEG server in the same container
# ---------------------------------------------------------------------------
class Broadcast:
    def __init__(self):
        self.jpeg = None
        self.n = 0
        self.viewers = 0
        self.last_seen = time.time()


def start_mjpeg_server(bc, fps):
    import asyncio
    import threading

    from aiohttp import web

    async def mjpeg(req):
        resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                           "Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"})
        await resp.prepare(req)
        bc.viewers += 1
        last = -1
        try:
            while True:
                if bc.n != last and bc.jpeg is not None:
                    jpg, last = bc.jpeg, bc.n
                    await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.5 / fps)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            bc.viewers -= 1
            bc.last_seen = time.time()
        return resp

    async def snapshot(req):
        if bc.jpeg is None:
            return web.Response(status=503, text="not ready")
        return web.Response(body=bc.jpeg, content_type="image/jpeg",
                            headers={"Access-Control-Allow-Origin": "*"})

    async def index(req):
        return web.Response(content_type="text/html", text=(
            '<body style="margin:0;background:#000;display:grid;place-items:center;height:100vh">'
            '<img src="stream.mjpg" style="max-width:100%;max-height:100vh"></body>'))

    web_app = web.Application()
    web_app.add_routes([web.get("/", index), web.get("/stream.mjpg", mjpeg), web.get("/frame.jpg", snapshot)])

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(web_app)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", PORT).start())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()


def _update(sid, **kw):
    s = sessions.get(sid, {})
    s.update(kw)
    sessions[sid] = s


@app.cls(image=gpu_image, gpu="L4", volumes={MOUNT: vol}, timeout=24 * 60 * 60, max_containers=MAX_STREAMS)
class Viewer:
    model: str = modal.parameter(default=PKL_NAME)

    @modal.enter()
    def load(self):
        import torch
        import legacy

        import glob
        path = glob.glob(f"{MOUNT}/**/{find_pkl(self.model)}", recursive=True)[0]  # partial names OK
        with open(path, "rb") as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].cuda().eval()
        G = self.G
        print(f"Loaded {self.model}: z_dim={G.z_dim} w_dim={G.w_dim} layers(num_ws)={G.num_ws} "
              f"res={G.img_resolution}")
        with torch.inference_mode():
            G(torch.zeros(1, G.z_dim, device="cuda"), torch.zeros(1, G.c_dim, device="cuda"), noise_mode="const")
            # W statistics: mean and per-dimension std over 10k random z, so amp is in "W std" units
            ws = []
            for _ in range(10):
                z = torch.randn(1000, G.z_dim, device="cuda")
                ws.append(G.mapping(z, torch.zeros(1000, G.c_dim, device="cuda"), truncation_psi=1)[:, 0])
            ws = torch.cat(ws)
            self.w_avg, self.w_std = ws.mean(0), ws.std(0)          # (w_dim,)
        vol.commit()

    @modal.method()
    def run(self, sid: str, p: dict):
        import io

        import numpy as np
        import torch
        import torch.nn.functional as F

        try:
            from torchvision.io import encode_jpeg
            encode_jpeg(torch.zeros(3, 16, 16, dtype=torch.uint8, device="cuda"))
            gpu_jpeg = True
        except Exception as e:  # older torchvision: encode on CPU instead
            print("GPU JPEG unavailable, using PIL:", e)
            gpu_jpeg = False

        G = self.G
        fps, size, quality = p["fps"], p["size"], p["quality"]
        c = torch.zeros(1, G.c_dim, device="cuda")
        layers = parse_layers(p["layers"], G.num_ws)
        wplus = p["space"] == "wplus"
        with torch.inference_mode():
            z0 = torch.from_numpy(np.random.RandomState(p["seed"]).randn(1, G.z_dim)).float().cuda()
            w_base = G.mapping(z0, c, truncation_psi=1)[0, 0]                        # (w_dim,)
        noise = PerlinNoise(p["seed"], G.w_dim * (len(layers) if wplus else 1), p["speed"])
        amp_std = (p["amp"] * self.w_std).float()                                    # (w_dim,)
        print(f"W walk: space={p['space']} layers={layers} amp={p['amp']} speed={p['speed']} psi={p['psi']}")
        fx = None
        if p.get("fluid", 0) > 0:
            out = size or G.img_resolution
            fx = FluidFX(out, out, p["seed"], strength=p["fluid"] * out / 512, mix=p["fluid_mix"],
                         scale=p["fluid_scale"], follow=p["fluid_follow"], flow_speed=p["fluid_speed"])
            print(f"Fluid: strength={p['fluid']} mix={p['fluid_mix']} scale={p['fluid_scale']} "
                  f"follow={p['fluid_follow']} speed={p['fluid_speed']}")

        bc = Broadcast()
        start_mjpeg_server(bc, fps)
        try:
            with modal.forward(PORT) as tunnel:
                t_start = time.time()
                _update(sid, status="live", gpu_url=tunnel.url, stream_url=f"{tunnel.url}/stream.mjpg",
                        live_at=t_start)
                print(f"\n========== WATCH ==========\n{tunnel.url}\n===========================\n", flush=True)
                bc.last_seen = t_start
                end_reason, i, last_check = None, 0, 0.0
                with torch.inference_mode():
                    while end_reason is None:
                        t0 = time.time()
                        off = torch.from_numpy(noise(np.array([i / fps]))[0]).float().cuda()
                        off = off.view(-1, G.w_dim) * amp_std                         # (1 or n_layers, w_dim)
                        ws = w_base.expand(G.num_ws, -1).clone()                      # (num_ws, w_dim)
                        ws[layers] = w_base + off                                      # moving layers
                        ws = self.w_avg + p["psi"] * (ws - self.w_avg)                # truncation
                        img = G.synthesis(ws[None], noise_mode="const")                # (1,3,H,W) [-1,1]
                        if size and size != img.shape[-1]:
                            img = F.interpolate(img, size=(size, size), mode="area" if size < img.shape[-1] else "bilinear")
                        if fx is not None:
                            img = fx(((img[0] + 1) / 2).clamp(0, 1), i / fps)              # fluid canvas, [0,1]
                            img = (img * 255 + 0.5).clamp(0, 255).to(torch.uint8)
                        else:
                            img = (img[0] * 127.5 + 128).clamp(0, 255).to(torch.uint8)   # (3,H,W) on GPU
                        if gpu_jpeg:
                            jpg = encode_jpeg(img, quality=quality).cpu().numpy().tobytes()  # only JPEG bytes leave the GPU
                        else:
                            from PIL import Image
                            buf = io.BytesIO()
                            Image.fromarray(img.permute(1, 2, 0).cpu().numpy()).save(buf, "JPEG", quality=quality)
                            jpg = buf.getvalue()
                        bc.jpeg, bc.n = jpg, bc.n + 1
                        i += 1

                        now = time.time()
                        if now - last_check > 1:
                            last_check = now
                            if sessions.get(f"{sid}:stop"):
                                end_reason = "stopped"
                            elif now - t_start > p["max_minutes"] * 60:
                                end_reason = "max_minutes reached"
                            elif bc.viewers == 0 and now - bc.last_seen > p["idle_minutes"] * 60:
                                end_reason = "no viewers"
                            if i % (fps * 30) < fps:
                                print(f"[{sid}] {i / (now - t_start):.1f} fps, {bc.viewers} viewers, "
                                      f"{len(jpg) / 1024:.0f} KB/frame")
                        time.sleep(max(0.0, 1 / fps - (time.time() - t0)))
            _update(sid, status="ended", reason=end_reason, ended_at=time.time())
            print(f"[{sid}] ended: {end_reason}")
        except Exception as e:
            _update(sid, status="error", error=repr(e))
            raise


# ---------------------------------------------------------------------------
# API (CPU): POST to start, GET status, DELETE to stop, /watch page for viewers
# ---------------------------------------------------------------------------
WATCH_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Live latent walk</title>
<style>:root{color-scheme:dark}body{margin:0;background:#000;color:#bbb;font:14px system-ui,sans-serif;
display:grid;place-items:center;min-height:100vh}img{max-width:100vw;max-height:100vh;display:none}
#s{position:fixed;bottom:12px;left:0;right:0;text-align:center;opacity:.7}</style></head>
<body><img id="v" alt="live"><div id="s">Starting GPU… (first start can take a minute)</div>
<script>
const id="__ID__", v=document.getElementById('v'), s=document.getElementById('s'); let shown=false;
async function poll(){
  try{
    const r=await fetch('../streams/'+id); if(!r.ok){s.textContent='Stream not found';return;}
    const d=await r.json();
    if(d.status==='live'){ if(!shown){v.src=d.stream_url; v.style.display='block'; shown=true; s.textContent='';} }
    else if(d.status==='ended'){v.style.display='none'; s.textContent='Stream ended ('+(d.reason||'')+')'; return;}
    else if(d.status==='error'){s.textContent='Error: '+d.error; return;}
    else s.textContent='Starting GPU… ('+d.status+')';
  }catch(e){}
  setTimeout(poll,2000);
}
v.onerror=()=>{shown=false; v.style.display='none'; s.textContent='Reconnecting…'};
poll();
</script></body></html>"""


@app.function(image=api_image, volumes={MOUNT: vol}, secrets=[modal.Secret.from_name("unsupervised")])
@modal.asgi_app()
def api():
    import secrets as pysecrets

    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    web = FastAPI(title="GPU latent viewer")

    class StreamRequest(BaseModel):
        model: str = PKL_NAME
        seed: int = 0
        speed: float = Field(0.15, gt=0, le=5)
        amp: float = Field(1.0, ge=0, le=10)
        space: str = Field("w", pattern="^(w|wplus)$")
        layers: str = Field("all", max_length=64)
        fluid: float = Field(2.0, ge=0, le=20)
        fluid_mix: float = Field(0.12, gt=0, le=1)
        fluid_scale: float = Field(4.0, ge=1, le=32)
        fluid_follow: float = Field(0.5, ge=0, le=1)
        fluid_speed: float = Field(0.5, gt=0, le=5)
        psi: float = Field(0.8, ge=0, le=2)
        fps: int = Field(20, ge=1, le=60)
        size: int = Field(512, ge=0, le=2048)
        quality: int = Field(80, ge=10, le=100)
        idle_minutes: float = Field(5, gt=0, le=240)
        max_minutes: float = Field(120, gt=0, le=1440)
        gpu: str = "L4"

    def check_key(key):
        if not key or not pysecrets.compare_digest(key, os.environ.get("API_KEY", "")):
            raise HTTPException(401, "missing or wrong X-API-Key")

    def active_count():
        try:
            return sum(1 for k, v in sessions.items()
                       if ":" not in k and isinstance(v, dict) and v.get("status") in ("starting", "live"))
        except Exception:
            return 0

    @web.get("/models")
    def models():
        vol.reload()
        return {"models": list_models()}

    @web.post("/streams")
    def create(req: StreamRequest, request: Request, x_api_key: str = Header(None)):
        check_key(x_api_key)
        if req.gpu not in GPUS:
            raise HTTPException(400, f"gpu must be one of {sorted(GPUS)}")
        try:
            parse_layers(req.layers, 18)  # syntax check; exact range is checked on the GPU
        except ValueError:
            raise HTTPException(400, f"layers '{req.layers}' invalid; use e.g. all, 0-3, 0,1,8-11")
        vol.reload()
        try:
            model = find_pkl(req.model)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if active_count() >= MAX_STREAMS:
            raise HTTPException(429, f"{MAX_STREAMS} streams already running; stop one first")

        sid = pysecrets.token_urlsafe(9)
        params = req.model_dump() | {"model": model}
        sessions[sid] = {"status": "starting", "params": params, "created": time.time()}
        call = Viewer.with_options(gpu=req.gpu)(model=model).run.spawn(sid, params)
        _update(sid, call_id=call.object_id)

        base = str(request.base_url).rstrip("/")
        return {"id": sid, "status": "starting", "watch_url": f"{base}/watch/{sid}",
                "status_url": f"{base}/streams/{sid}", "params": params}

    @web.get("/streams/{sid}")
    def status(sid: str):
        s = sessions.get(sid)
        if not s:
            raise HTTPException(404, "no such stream")
        return {"id": sid, **{k: v for k, v in s.items() if k != "call_id"}}

    @web.delete("/streams/{sid}")
    def stop(sid: str, x_api_key: str = Header(None)):
        check_key(x_api_key)
        s = sessions.get(sid)
        if not s:
            raise HTTPException(404, "no such stream")
        sessions[f"{sid}:stop"] = True
        if s.get("status") == "starting" and s.get("call_id"):  # not live yet: cancel the GPU job
            modal.FunctionCall.from_id(s["call_id"]).cancel()
            _update(sid, status="ended", reason="stopped")
        return {"id": sid, "stopping": True}

    @web.get("/watch/{sid}", response_class=HTMLResponse)
    def watch(sid: str):
        if not sessions.get(sid):
            raise HTTPException(404, "no such stream")
        return WATCH_HTML.replace("__ID__", sid)

    return web


# ---------------------------------------------------------------------------
# Run one stream straight from the terminal, no API needed:
#   modal run gpu_view_w.py --model ffhq --seed 42 --space wplus --layers 0-3
# Prints a WATCH URL; Ctrl+C stops it (it also stops after --idle-minutes with no viewers).
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(model: str = PKL_NAME, seed: int = 0, speed: float = 0.15, amp: float = 1.0, space: str = "w",
         fluid: float = 2.0, fluid_mix: float = 0.12, fluid_scale: float = 4.0, fluid_follow: float = 0.5,
         fluid_speed: float = 0.5,
         layers: str = "all", psi: float = 0.8, fps: int = 20, size: int = 512, quality: int = 80,
         idle_minutes: float = 30, max_minutes: float = 120, gpu: str = "L4"):
    import secrets as pysecrets
    if space not in ("w", "wplus"):
        raise SystemExit("--space must be w or wplus")
    if gpu not in GPUS:
        raise SystemExit(f"--gpu must be one of {sorted(GPUS)}")
    parse_layers(layers, 18)
    sid = "local-" + pysecrets.token_urlsafe(6)
    params = dict(model=model, seed=seed, speed=speed, amp=amp, space=space, layers=layers, psi=psi, fps=fps,
                  fluid=fluid, fluid_mix=fluid_mix, fluid_scale=fluid_scale, fluid_follow=fluid_follow,
                  fluid_speed=fluid_speed,
                  size=size, quality=quality, idle_minutes=idle_minutes, max_minutes=max_minutes, gpu=gpu)
    sessions[sid] = {"status": "starting", "params": params, "created": time.time()}
    print(f"Starting W walk on {model} ({gpu}, first start can take a minute)... Ctrl+C to stop.")
    Viewer.with_options(gpu=gpu)(model=model).run.remote(sid, params)