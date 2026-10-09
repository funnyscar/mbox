"""
StyleGAN2 latent walks viewed straight off the GPU. No Cloudflare.

POST to the API -> a GPU container starts rendering -> you get a link that plays
frames JPEG-encoded on the GPU and served from that same container (MJPEG).

Setup (once):
  modal secret create gpu-view API_KEY=$(openssl rand -hex 16)     # protects POST/DELETE
Deploy:
  modal deploy gpu_view.py          # prints the API URL, e.g. https://<you>--gpu-view-api.modal.run

Use:
  curl -X POST "$API/streams" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
       -d '{"model": "ffhq", "seed": 42, "mode": "perlin"}'
    -> {"id": "...", "watch_url": "$API/watch/<id>", ...}    share watch_url with viewers
  curl "$API/streams/<id>"                                    status + direct GPU URL
  curl -X DELETE "$API/streams/<id>" -H "X-API-Key: $KEY"     stop it
  curl "$API/models"                                          models in the volume

POST body (all optional):
  model         .pkl in the "anadol" volume; partial names work     (default WikiArt_Uncond2)
  seed, mode    "perlin" | "walk"                                    (default 0, perlin)
  speed, amp    walk speed / perlin drift                            (default 0.3, 1.5)
  psi           truncation                                           (default 0.8)
  fps           frames per second                                    (default 20)
  size          output px (square), 0 = model resolution             (default 512)
  quality       JPEG quality 1-100                                   (default 80)
  idle_minutes  stop after this long with no viewers                 (default 5)
  max_minutes   hard stop                                            (default 120)
  gpu           "L4" | "A10G" | "L40S" | "T4"                        (default L4)
"""
import os
import time

import modal

VOLUME_NAME = "anadol"
PKL_NAME = "WikiArt_Uncond2.pkl"
MOUNT = "/anadol"
PORT = 8000
MAX_STREAMS = 3                      # max simultaneous GPU containers
GPUS = {"L4", "A10G", "L40S", "T4"}

app = modal.App("unsupervised")
vol = modal.Volume.from_name(VOLUME_NAME)
sessions = modal.Dict.from_name("unsupervised-sessions", create_if_missing=True)

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
# Endless latent paths (same as stream.py). Each z is rescaled to norm sqrt(z_dim).
# ---------------------------------------------------------------------------
def _renorm(z):
    import numpy as np
    return z / np.linalg.norm(z, axis=-1, keepdims=True) * np.sqrt(z.shape[-1])


class PerlinZ:
    def __init__(self, seed, z_dim, speed, amp, octaves=2):
        import numpy as np
        rng = np.random.RandomState(seed)
        self.base = rng.randn(z_dim)
        self.dims = np.arange(z_dim, dtype=np.uint64)[None, :]
        self.octaves = [(speed * 2 ** o, 0.5 ** o, rng.rand(1, z_dim) * 100, np.uint64(rng.randint(1, 2**31)))
                        for o in range(octaves)]
        self.scale = amp * 2 * np.sqrt(3)

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
        path = glob.glob(f"{MOUNT}/**/{self.model}", recursive=True)[0]
        with open(path, "rb") as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].cuda().eval()
        print(f"Loaded {self.model}: z_dim={self.G.z_dim} res={self.G.img_resolution}")
        with torch.inference_mode():
            self.G(torch.zeros(1, self.G.z_dim, device="cuda"),
                   torch.zeros(1, self.G.c_dim, device="cuda"), noise_mode="const")
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
        path = (PerlinZ(p["seed"], G.z_dim, p["speed"], p["amp"]) if p["mode"] == "perlin"
                else WalkZ(p["seed"], G.z_dim, p["speed"]))
        c = torch.zeros(1, G.c_dim, device="cuda")

        bc = Broadcast()
        start_mjpeg_server(bc, fps)
        try:
            with modal.forward(PORT) as tunnel:
                t_start = time.time()
                _update(sid, status="live", gpu_url=tunnel.url, stream_url=f"{tunnel.url}/stream.mjpg",
                        live_at=t_start)
                print(f"[{sid}] live at {tunnel.url}")
                bc.last_seen = t_start
                end_reason, i, last_check = None, 0, 0.0
                with torch.inference_mode():
                    while end_reason is None:
                        t0 = time.time()
                        z = torch.from_numpy(path(np.array([i / fps]))).float().cuda()
                        img = G(z, c, truncation_psi=p["psi"], noise_mode="const")       # (1,3,H,W) [-1,1]
                        if size and size != img.shape[-1]:
                            img = F.interpolate(img, size=(size, size), mode="area" if size < img.shape[-1] else "bilinear")
                        img = (img[0] * 127.5 + 128).clamp(0, 255).to(torch.uint8)       # (3,H,W) on GPU
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
        mode: str = Field("perlin", pattern="^(perlin|walk)$")
        speed: float = Field(0.3, gt=0, le=5)
        amp: float = Field(1.5, ge=0, le=10)
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
#   modal run gpu_view.py --model ffhq --seed 42
# Prints a WATCH URL; Ctrl+C stops it (it also stops after --idle-minutes with no viewers).
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(model: str = PKL_NAME, seed: int = 0, mode: str = "perlin", speed: float = 0.3, amp: float = 1.5,
         psi: float = 0.8, fps: int = 20, size: int = 512, quality: int = 80, idle_minutes: float = 30,
         max_minutes: float = 120, gpu: str = "L4"):
    import secrets as pysecrets
    if mode not in ("perlin", "walk"):
        raise SystemExit("--mode must be perlin or walk")
    if gpu not in GPUS:
        raise SystemExit(f"--gpu must be one of {sorted(GPUS)}")
    sid = "local-" + pysecrets.token_urlsafe(6)
    params = dict(model=model, seed=seed, mode=mode, speed=speed, amp=amp, psi=psi, fps=fps, size=size,
                  quality=quality, idle_minutes=idle_minutes, max_minutes=max_minutes, gpu=gpu)
    sessions[sid] = {"status": "starting", "params": params, "created": time.time()}
    print(f"Starting {model} on {gpu} (first start can take a minute)... Ctrl+C to stop.")
    Viewer.with_options(gpu=gpu)(model=model).run.remote(sid, params)
 