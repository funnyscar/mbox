"""
Minimal RTMPS test stream: white screen + white noise -> Cloudflare Stream.
No GPU, no model. Use it to check the Cloudflare side on its own.

Run on your computer (needs ffmpeg; reads .env next to this file):
  python3 simple_streams.py
  python3 simple_streams.py --minutes 2

Run on Modal (uses the "cloudflare-stream" secret; tests from Modal's network):
  modal run simple_streams.py
  modal run simple_streams.py --minutes 2

While it runs, check:  https://customer-<CODE>.cloudflarestream.com/<LIVE_INPUT_ID>/lifecycle?t=1
"""
import os
import subprocess
import sys


def load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_stream(minutes: float = 0):
    url = os.environ["CF_RTMPS_URL"].strip()
    key = os.environ.get("CF_STREAM_KEY", "").strip()
    if not url.endswith("/"):
        url += "/"
    dst = url + key if key else url
    print(f"Streaming to {url}...{key[-6:]}  ({'until stopped' if not minutes else f'{minutes} min'})")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-stats",
        "-re", "-f", "lavfi", "-i", "color=c=white:s=1280x720:r=30",
        "-re", "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.1:sample_rate=48000",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
        "-b:v", "1500k", "-maxrate", "1500k", "-bufsize", "3000k", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0", "-bf", "0",
        "-force_key_frames", "expr:gte(t,n_forced*2)",  # IDR keyframe exactly every 2 s
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
    ]
    if minutes:
        cmd += ["-t", str(minutes * 60)]
    cmd += ["-f", "flv", dst]
    rc = subprocess.call(cmd)
    print("ffmpeg exit code:", rc)
    return rc


# ---------------- Modal (optional) ----------------
try:
    import modal

    app = modal.App("simple-stream")
    image = modal.Image.debian_slim().apt_install("ffmpeg")

    @app.function(image=image, secrets=[modal.Secret.from_name("cloudflare-stream")], timeout=24 * 3600)
    def stream_remote(minutes: float = 0):
        return run_stream(minutes)

    @app.local_entrypoint()
    def main(minutes: float = 0):
        stream_remote.remote(minutes)
except ImportError:
    pass


if __name__ == "__main__":
    load_env()
    mins = 0.0
    if "--minutes" in sys.argv:
        mins = float(sys.argv[sys.argv.index("--minutes") + 1])
    if not os.environ.get("CF_RTMPS_URL"):
        sys.exit("Missing CF_RTMPS_URL (put it in .env)")
    sys.exit(run_stream(mins))