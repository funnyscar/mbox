"""Render a MediaPipe face mesh over a video, on Modal.

Using: 

Usage:
    modal run video.py --src clip.mp4 [--dst out.mp4]
"""
from pathlib import Path

import modal

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "/models/face_landmarker.task"

image = (
    modal.Image.debian_slim(python_version="3.11")
	.apt_install("libgl1", "libegl1", "libgles2", "libglib2.0-0", "wget", "ffmpeg")
    .pip_install("mediapipe", "opencv-python-headless", "numpy")
    .run_commands(f"mkdir -p /models && wget -q -O {MODEL_PATH} {MODEL_URL}")
)

app = modal.App("face-mesh-video", image=image)


@app.function(cpu=4, memory=4096, timeout=60 * 60)
def render(video_bytes: bytes, suffix: str = ".mp4") -> tuple[bytes, int]:
    import subprocess
    import tempfile

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    tmp = Path(tempfile.mkdtemp())
    src, raw, out = tmp / f"in{suffix}", tmp / "raw.mp4", tmp / "out.mp4"
    src.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=5,
    )
    C = vision.FaceLandmarksConnections
    layers = [  # (connections, BGR color, thickness)
        (C.FACE_LANDMARKS_TESSELATION, (180, 180, 180), 1),
        (C.FACE_LANDMARKS_CONTOURS, (255, 255, 255), 1),
        (C.FACE_LANDMARKS_LEFT_IRIS, (0, 255, 255), 1),
        (C.FACE_LANDMARKS_RIGHT_IRIS, (0, 255, 255), 1),
    ]

    frames = 0
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, int(frames * 1000 / fps))
            for face in result.face_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in face]
                for conns, color, thick in layers:
                    for c in conns:
                        cv2.line(bgr, pts[c.start], pts[c.end], color, thick, cv2.LINE_AA)
            writer.write(bgr)
            frames += 1
    cap.release()
    writer.release()

    # Re-encode to H.264 (plays everywhere) and carry over original audio if present.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(src),
         "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )
    return out.read_bytes(), frames


@app.local_entrypoint()
def main(src: str, dst: str = ""):
    src_p = Path(src)
    dst_p = Path(dst) if dst else src_p.with_name(src_p.stem + "_mesh.mp4")
    out, n = render.remote(src_p.read_bytes(), src_p.suffix or ".mp4")
    dst_p.write_bytes(out)
    print(f"{n} frames -> {dst_p}")