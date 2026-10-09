"""Render a MediaPipe face mesh over an image, on Modal.

Usage:
    modal run single.py --src photo.jpg [--dst out.jpg]
"""
from pathlib import Path

import modal

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "/models/face_landmarker.task"

image = (
    modal.Image.debian_slim(python_version="3.11")
	.apt_install("libgl1", "libegl1", "libgles2", "libglib2.0-0", "wget")
    .pip_install("mediapipe", "opencv-python-headless", "numpy")
    .run_commands(f"mkdir -p /models && wget -q -O {MODEL_PATH} {MODEL_URL}")
)

app = modal.App("face-mesh-overlay", image=image)


@app.function()
def render(img_bytes: bytes, ext: str = ".jpg") -> tuple[bytes, int]:
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=5,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_img)

    C = vision.FaceLandmarksConnections
    layers = [  # (connections, BGR color, thickness)
        (C.FACE_LANDMARKS_TESSELATION, (180, 180, 180), 1),
        (C.FACE_LANDMARKS_CONTOURS, (255, 255, 255), 1),
        (C.FACE_LANDMARKS_LEFT_IRIS, (0, 255, 255), 1),
        (C.FACE_LANDMARKS_RIGHT_IRIS, (0, 255, 255), 1),
    ]
    h, w = bgr.shape[:2]
    for face in result.face_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in face]
        for conns, color, thick in layers:
            for c in conns:
                cv2.line(bgr, pts[c.start], pts[c.end], color, thick, cv2.LINE_AA)

    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError(f"Could not encode output as {ext}")
    return buf.tobytes(), len(result.face_landmarks)


@app.local_entrypoint()
def main(src: str, dst: str = ""):
    src_p = Path(src)
    dst_p = Path(dst) if dst else src_p.with_stem(src_p.stem + "_mesh")
    out, n = render.remote(src_p.read_bytes(), dst_p.suffix or ".jpg")
    dst_p.write_bytes(out)
    print(f"{n} face(s) -> {dst_p}")