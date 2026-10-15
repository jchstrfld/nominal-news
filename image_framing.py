"""Local face-aware positioning for the reader's existing 16:9 image frame.

This module never selects images or decides whether their licenses are acceptable.
Call only after the existing image rights check. No remote inference is used.
"""
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/face_detection/yunet.onnx"
CACHE = ROOT / "data/image_framing"
VERSION = "yunet-2023mar-v1"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
ALLOWED_HOSTS = {"upload.wikimedia.org", "images.unsplash.com"}
MAX_BYTES = 12 * 1024 * 1024


def centered(reason):
    return {"x": 50.0, "y": 50.0, "reason": reason, "version": VERSION}


def position_for_faces(width, height, faces):
    """Keep a clear largest face, with headroom, inside the cover crop.

    CSS percentages refer to the *overflow*, not to image coordinates.
    Similar-sized faces are ambiguous: leave the existing crop alone.
    """
    valid = []
    for row in faces:
        x, y, w, h, score = map(float, row)
        if (all(math.isfinite(v) for v in (x, y, w, h, score))
                and score >= .9 and min(w, h) >= 12
                and x >= 0 and y >= 0 and x + w <= width and y + h <= height):
            valid.append((x, y, w, h, score))
    valid.sort(key=lambda f: f[2] * f[3], reverse=True)
    if not valid:
        return centered("no_clear_face")
    if len(valid) > 1 and valid[0][2] * valid[0][3] < 2 * valid[1][2] * valid[1][3]:
        return centered("ambiguous_faces")
    x, y, w, h, _ = valid[0]
    left, right = max(0, x - .2 * w), min(width, x + 1.2 * w)
    top, bottom = max(0, y - .4 * h), min(height, y + 1.15 * h)
    crop_w, crop_h = min(width, height * 16 / 9), min(height, width * 9 / 16)
    if right - left > crop_w or bottom - top > crop_h:
        return centered("face_exceeds_frame")

    def axis(size, crop, lo, hi):
        overflow = size - crop
        if overflow <= .001:
            return 50.0
        # Keep the existing crop if it already preserves the padded face.
        if overflow / 2 <= lo and overflow / 2 + crop >= hi:
            return 50.0
        start = max(0, min(overflow, (lo + hi - crop) / 2))
        return round(100 * start / overflow, 3)

    return {"x": axis(width, crop_w, left, right),
            "y": axis(height, crop_h, top, bottom),
            "reason": "primary_face", "version": VERSION}


def analyze_image(content):
    if len(content) > MAX_BYTES:
        raise ValueError("image_too_large")
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("unexpected_detector_model")
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid_image")
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(width, height))
    image = cv2.resize(image, (round(width * scale), round(height * scale)))
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(MODEL), "", (width, height), .9, .3, 5000)
    _, faces = detector.detect(image)
    boxes = [] if faces is None else [list(f[:4]) + [f[-1]] for f in faces]
    return position_for_faces(width, height, boxes)


def download_image(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("unsupported_image_host")
    # No redirects to arbitrary destinations, retries, or paid services.
    with requests.get(url, timeout=(5, 15), stream=True, allow_redirects=False,
                      headers={"User-Agent": "NominalNews/1.0 (image framing)"}) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("image_redirect")
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError("image_too_large")
            chunks.append(chunk)
    return b"".join(chunks)


def frame_image(url, *, cache_dir=CACHE, content=None):
    """Cache successful analysis; failures retain center and can retry next run.

    `content` supports offline replay of the exact selected photo.
    Cache contains only position metadata, never photos or face embeddings.
    """
    if not url:
        return centered("no_image")
    key = hashlib.sha256((VERSION + url).encode()).hexdigest()
    path = Path(cache_dir) / (key + ".json")
    if content is None:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (cached.get("version") == VERSION and
                    all(isinstance(cached.get(k), (int, float)) and
                        0 <= cached[k] <= 100 for k in ("x", "y"))):
                return cached
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    try:
        result = analyze_image(content if content is not None else download_image(url))
    except (OSError, ValueError, requests.RequestException, cv2.error):
        return centered("analysis_unavailable")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass  # Framing should not fail the daily pipeline over a cache write.
    return result
