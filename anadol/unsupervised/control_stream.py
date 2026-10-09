"""
Interactive StyleGAN2 stream: steer chosen z dimensions with the arrow keys, live to Cloudflare Stream.

  Left/Right  -> moves along --x-dims        Up/Down -> moves along --y-dims
  Shift       -> 3x faster                   R       -> back to the start

Run (uses the same "cloudflare-stream" secret and "anadol" volume as stream.py):
  modal run --detach control_stream.py --model ffhq --x-dims 12 --y-dims 40
  modal run --detach control_stream.py --model modern_art --x-dims 0,1,2 --y-dims 100,101 --seed 7 --speed 1.5
Then open the CONTROL URL printed in the logs (modal app logs anadol-control).
The control page shows a low-latency preview straight from the GPU; the Cloudflare
stream that viewers watch is ~5-15 s behind.
Stop:
  modal app stop anadol-control

Options:
  --model     .pkl in the volume; partial names work (e.g. ffhq, cat, modern_art)
  --x-dims    z indices moved by Left/Right, comma-separated (default 0)
  --y-dims    z indices moved by Up/Down, comma-separated (default 1)
  --seed      starting z (default 0)
  --speed     z units per second while a key is held (default 1.0)
  --range     max offset from the start along each axis (default 4.0)
  --psi       truncation (default 0.8)
  --fps       stream frame rate (default 30)
  --size      output width/height in px, 0 = model resolution (default 0)
  --bitrate   video bitrate (default 6M)
  --minutes   stop after N minutes, 0 = run until stopped (max 24 h)
  --gpu       Modal GPU type (default L4; needs NVENC: L4, T4, A10G, L40S)
"""
import os

import modal

VOLUME_NAME = "anadol"
PKL_NAME = "WikiArt_Uncond2.pkl"
MOUNT = "/anadol"
PORT = 8000


def find_pkl(name: str) -> str:
    """Find a .pkl in the volume by exact name or unique partial match (case-insensitive)."""
    import glob
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
    .pip_install("torch", "numpy", "pillow", "click", "requests", "scipy", "ninja", "aiohttp",
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
        "NVIDIA_DRIVER_CAPABILITIES": "all",
    })
)

app = modal.App("anadol-control", image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


# ---------------------------------------------------------------------------
# Control state + motion
# ---------------------------------------------------------------------------
class Control:
    """Shared between the web server thread and the render thread."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.keys = set()        # held: "left" "right" "up" "down"
        self.fast = False
        self.reset = False
        self.clients = 0
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.jpeg = None         # latest preview frame
        self.frame = None        # latest full frame (raw RGB bytes) for ffmpeg

    def press(self, key, down, fast=False):
        with self.lock:
            if key == "reset":
                self.reset = bool(down)
                return
            if key in ("left", "right", "up", "down"):
                (self.keys.add if down else self.keys.discard)(key)
            self.fast = bool(fast)

    def release_all(self):
        with self.lock:
            self.keys.clear()
            self.fast = False


def step_motion(ctrl, dt, speed, rng, accel=8.0):
    """Advance x/y by one time step: velocity eases toward the held-key target, clamped to +-rng."""
    with ctrl.lock:
        keys, fast = set(ctrl.keys), ctrl.fast
        if ctrl.reset:
            ctrl.x = ctrl.y = ctrl.vx = ctrl.vy = 0.0
            ctrl.reset = False
        s = speed * (3.0 if fast else 1.0)
        tx = s * (("right" in keys) - ("left" in keys))
        ty = s * (("up" in keys) - ("down" in keys))
        a = min(1.0, dt * accel)
        ctrl.vx += (tx - ctrl.vx) * a
        ctrl.vy += (ty - ctrl.vy) * a
        ctrl.x = max(-rng, min(rng, ctrl.x + ctrl.vx * dt))
        ctrl.y = max(-rng, min(rng, ctrl.y + ctrl.vy * dt))
        return ctrl.x, ctrl.y


def make_z(base, x_dims, y_dims, x, y):
    z = base.copy()
    z[x_dims] = base[x_dims] + x
    z[y_dims] = base[y_dims] + y
    return z


def parse_dims(s, z_dim=None, name="dims"):
    dims = [int(p) for p in str(s).replace(" ", "").split(",") if p != ""]
    if not dims:
        raise ValueError(f"--{name} is empty")
    if z_dim is not None:
        bad = [d for d in dims if not 0 <= d < z_dim]
        if bad:
            raise ValueError(f"--{name} {bad} out of range: this model's z has indices 0..{z_dim - 1}")
    return dims


# ---------------------------------------------------------------------------
# Web control page (arrow keys / touch buttons) + MJPEG preview, served via modal.forward
# ---------------------------------------------------------------------------
CONTROL_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Latent control</title>
<style>
 :root{color-scheme:dark} body{margin:0;background:#0b0b0c;color:#ddd;font:14px system-ui,sans-serif;
 display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
 img{max-width:min(92vw,640px);border-radius:8px;background:#000}
 #s{font-variant-numeric:tabular-nums;opacity:.85} .pad{display:grid;grid-template-columns:repeat(3,64px);gap:6px}
 button{height:64px;font-size:22px;border:0;border-radius:10px;background:#222;color:#eee;touch-action:none}
 button.on{background:#3a6df0} a{color:#8ab4ff}
</style></head><body>
<div id="s">connecting…</div>
<img src="preview.mjpg?k=__KEY__" alt="preview">
<div class="pad">
 <span></span><button data-k="up">↑</button><span></span>
 <button data-k="left">←</button><button data-k="reset">R</button><button data-k="right">→</button>
 <span></span><button data-k="down">↓</button><span></span>
</div>
<div>Arrow keys or buttons · Shift = fast · R = reset</div>
__PLAYER__
<script>
const K="__KEY__", s=document.getElementById('s'); let ws;
const map={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'up',ArrowDown:'down',r:'reset',R:'reset'};
function send(k,d,f){ if(ws&&ws.readyState===1) ws.send(JSON.stringify({k,d:d?1:0,f:f?1:0})); }
function connect(){
  ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws?k='+K);
  ws.onopen=()=>s.textContent='connected';
  ws.onmessage=e=>{const m=JSON.parse(e.data); s.textContent=`x ${m.x.toFixed(2)}   y ${m.y.toFixed(2)}`;};
  ws.onclose=()=>{s.textContent='disconnected – retrying…'; setTimeout(connect,1000);};
}
connect();
const held=new Set();
addEventListener('keydown',e=>{const k=map[e.key]; if(!k) return; e.preventDefault();
  if(!held.has(k)||e.shiftKey){held.add(k); send(k,1,e.shiftKey);} });
addEventListener('keyup',e=>{const k=map[e.key]; if(!k) return; held.delete(k); send(k,0,e.shiftKey);});
addEventListener('blur',()=>{held.forEach(k=>send(k,0,0)); held.clear();});
document.querySelectorAll('button').forEach(b=>{const k=b.dataset.k;
  b.addEventListener('pointerdown',e=>{b.classList.add('on'); b.setPointerCapture(e.pointerId); send(k,1,0);});
  ['pointerup','pointercancel'].forEach(t=>b.addEventListener(t,()=>{b.classList.remove('on'); send(k,0,0);}));
});
</script></body></html>"""


def start_control_server(ctrl, token, player_url=""):
    import asyncio
    import json
    import threading

    from aiohttp import WSMsgType, web

    def authed(req):
        return req.query.get("k") == token

    async def index(req):
        if not authed(req):
            return web.Response(status=403, text="wrong or missing ?k= key")
        player = (f'<div>Public stream (delayed): <a href="{player_url}" target="_blank">{player_url}</a></div>'
                  if player_url else "")
        html = CONTROL_HTML.replace("__KEY__", token).replace("__PLAYER__", player)
        return web.Response(text=html, content_type="text/html")

    async def ws_handler(req):
        if not authed(req):
            return web.Response(status=403)
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(req)
        ctrl.clients += 1

        async def push_state():
            while not ws.closed:
                await ws.send_str(json.dumps({"x": ctrl.x, "y": ctrl.y}))
                await asyncio.sleep(0.1)

        pusher = asyncio.ensure_future(push_state())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        d = json.loads(msg.data)
                        ctrl.press(str(d.get("k")), d.get("d"), d.get("f"))
                    except (ValueError, AttributeError):
                        pass
        finally:
            pusher.cancel()
            ctrl.clients -= 1
            ctrl.release_all()  # never leave a key stuck down when a client drops
        return ws

    async def preview(req):
        if not authed(req):
            return web.Response(status=403)
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame", "Cache-Control": "no-cache"})
        await resp.prepare(req)
        last = None
        try:
            while True:
                jpg = ctrl.jpeg
                if jpg is not None and jpg is not last:
                    await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    last = jpg
                await asyncio.sleep(1 / 20)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    web_app = web.Application()
    web_app.add_routes([web.get("/", index), web.get("/ws", ws_handler), web.get("/preview.mjpg", preview)])

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(web_app)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", PORT).start())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------
# Encoding (same settings as stream.py)
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
           "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
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
            "-force_key_frames", "expr:gte(t,n_forced*2)",
            "-c:a", "aac", "-b:a", "128k", "-f", "flv", dst]
    return cmd


# ---------------------------------------------------------------------------
@app.cls(gpu="L4", volumes={MOUNT: vol}, secrets=[modal.Secret.from_name("anadol-stream")],
         timeout=24 * 60 * 60)
class ControlStreamer:
    model: str = modal.parameter(default=PKL_NAME)

    @modal.enter()
    def load(self):
        import torch
        import legacy

        path = find_pkl(self.model)
        with open(path, "rb") as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].cuda().eval()
        print(f"Loaded {path}: z_dim={self.G.z_dim} res={self.G.img_resolution}")
        with torch.inference_mode():
            self.G(torch.zeros(1, self.G.z_dim, device="cuda"),
                   torch.zeros(1, self.G.c_dim, device="cuda"), noise_mode="const")
        vol.commit()

    @modal.method()
    def stream(self, x_dims: str, y_dims: str, seed: int, speed: float, rng: float, psi: float,
               fps: int, size: int, bitrate: str, minutes: float):
        import io
        import secrets
        import subprocess
        import threading
        import time

        import numpy as np
        import torch
        from PIL import Image

        G = self.G
        xd = parse_dims(x_dims, G.z_dim, "x-dims")
        yd = parse_dims(y_dims, G.z_dim, "y-dims")
        base = np.random.RandomState(seed).randn(G.z_dim)
        res = G.img_resolution
        print(f"x-dims={xd}  y-dims={yd}  seed={seed}  speed={speed}  range=±{rng}")

        url = os.environ["CF_RTMPS_URL"].strip()
        url = url if url.endswith("/") else url + "/"
        dst = url + os.environ.get("CF_STREAM_KEY", "").strip()
        player = ""
        if os.environ.get("CF_CUSTOMER_CODE") and os.environ.get("CF_LIVE_INPUT_ID"):
            player = "https://customer-{}.cloudflarestream.com/{}/iframe".format(
                os.environ["CF_CUSTOMER_CODE"], os.environ["CF_LIVE_INPUT_ID"])

        ctrl = Control()
        token = os.environ.get("CONTROL_KEY") or secrets.token_urlsafe(9)
        start_control_server(ctrl, token, player)

        stop = threading.Event()
        err = []

        # Render thread: one frame per tick at most `fps`, driven by the current x/y
        def render():
            try:
                c = torch.zeros(1, G.c_dim, device="cuda")
                t_last, n, t0 = time.time(), 0, time.time()
                with torch.inference_mode():
                    while not stop.is_set():
                        now = time.time()
                        x, y = step_motion(ctrl, now - t_last, speed, rng)
                        t_last = now
                        z = torch.from_numpy(make_z(base, xd, yd, x, y)[None]).float().cuda()
                        img = G(z, c, truncation_psi=psi, noise_mode="const")
                        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                        ctrl.frame = img.tobytes()
                        if n % 2 == 0:  # low-latency preview for the control page, ~15 fps
                            im = Image.fromarray(img)
                            im.thumbnail((512, 512))
                            buf = io.BytesIO()
                            im.save(buf, format="JPEG", quality=75)
                            ctrl.jpeg = buf.getvalue()
                        n += 1
                        if n % (fps * 30) == 0:
                            print(f"render {n / (time.time() - t0):.1f} fps | x={ctrl.x:+.2f} y={ctrl.y:+.2f} "
                                  f"| clients={ctrl.clients}")
                        time.sleep(max(0.0, 1 / fps - (time.time() - now)))
            except Exception as e:
                import traceback
                traceback.print_exc()
                err.append(e)
                stop.set()

        threading.Thread(target=render, daemon=True).start()

        with modal.forward(PORT) as tunnel:
            print("\n========== CONTROL URL ==========")
            print(f"{tunnel.url}/?k={token}")
            print("=================================\n", flush=True)

            while ctrl.frame is None and not stop.is_set():
                time.sleep(0.1)

            encoder = _pick_encoder()
            end = time.time() + minutes * 60 if minutes > 0 else float("inf")
            sent = 0
            try:
                while time.time() < end and not stop.is_set():
                    print(f"Connecting to Cloudflare ({encoder}, {fps} fps)...")
                    enc = subprocess.Popen(_ffmpeg_cmd(res, size, fps, bitrate, encoder, dst),
                                           stdin=subprocess.PIPE)
                    t0, n = time.time(), 0
                    try:
                        # Writer: send the latest frame every tick (repeats it if the GPU is slower)
                        while time.time() < end and not stop.is_set():
                            delay = t0 + n / fps - time.time()
                            if delay > 0:
                                time.sleep(delay)
                            elif delay < -1:
                                t0, n = time.time(), 0
                            enc.stdin.write(ctrl.frame)
                            n += 1
                            sent += 1
                        break
                    except (BrokenPipeError, OSError):
                        print("Cloudflare connection lost; reconnecting in 3 s...")
                        time.sleep(3)
                    finally:
                        try:
                            enc.stdin.close()
                            enc.wait(timeout=10)
                        except Exception:
                            enc.kill()
            finally:
                stop.set()
            if err:
                raise RuntimeError("Renderer crashed (traceback above)") from err[0]
            print(f"Stream ended after {sent} frames.")


@app.local_entrypoint()
def main(model: str = PKL_NAME, x_dims: str = "0", y_dims: str = "1", seed: int = 0, speed: float = 1.0,
         range: float = 4.0, psi: float = 0.8, fps: int = 30, size: int = 0, bitrate: str = "6M",
         minutes: float = 0, gpu: str = "L4"):
    parse_dims(x_dims, name="x-dims")
    parse_dims(y_dims, name="y-dims")
    ControlStreamer.with_options(gpu=gpu)(model=model).stream.remote(
        x_dims, y_dims, seed, speed, range, psi, fps, size, bitrate, minutes)