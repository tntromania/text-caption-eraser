#!/usr/bin/env python3
"""
RunPod worker — text/logo removal din video
  Detectie  : Florence-2 (florence-community/Florence-2-large)
              Integrare nativa transformers, NO trust_remote_code
  Inpainting: ProPainter (video-aware, no flickering)
"""
import os, sys, subprocess, traceback, time, shutil, json

print("[INIT] Python start...", flush=True)

try:
    import base64, tempfile, requests
    import numpy as np, cv2, runpod
    from PIL import Image
    print("[INIT] Importuri de baza OK", flush=True)
except Exception as e:
    print(f"[FATAL] {e}", flush=True); traceback.print_exc(); sys.exit(1)

# ─────────────────────────────────────────────────────────────
# GPU init
# ─────────────────────────────────────────────────────────────
try:
    import torch
    USE_CUDA = torch.cuda.is_available()
    DEVICE   = torch.device('cuda' if USE_CUDA else 'cpu')
    if USE_CUDA:
        gpu_name = torch.cuda.get_device_name(0)
        cap      = torch.cuda.get_device_capability(0)
        print(f"[INIT] GPU: {gpu_name} sm_{cap[0]}{cap[1]}", flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        print("[INIT] Niciun GPU CUDA → CPU (lent!)", flush=True)
except Exception as e:
    print(f"[FATAL] torch: {e}", flush=True); traceback.print_exc(); sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Florence-2 — incarcata o singura data
# Folosim florence-community/Florence-2-large:
#   - integrare nativa in transformers (fara trust_remote_code)
#   - compatibil cu transformers 4.x si 5.x
# ─────────────────────────────────────────────────────────────
_FLORENCE_MODEL     = None
_FLORENCE_PROCESSOR = None
FLORENCE_MODEL_ID   = "florence-community/Florence-2-large"

def get_florence():
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    if _FLORENCE_MODEL is None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        print(f"[INIT] Incarc Florence-2 ({FLORENCE_MODEL_ID})...", flush=True)

        _FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID)

        _FLORENCE_MODEL = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE_MODEL_ID,
            torch_dtype=torch.float16 if USE_CUDA else torch.float32,
        ).to(DEVICE)
        _FLORENCE_MODEL.eval()
        print("[INIT] Florence-2 ready!", flush=True)
    return _FLORENCE_MODEL, _FLORENCE_PROCESSOR


def detect_with_florence(frame_rgb: np.ndarray) -> list[dict]:
    """
    Ruleaza Florence-2 cu OPEN_VOCABULARY_DETECTION pe un frame RGB.
    Returneaza: [{'x', 'y', 'w', 'h', 'label'}, ...]
    """
    model, processor = get_florence()
    pil_img = Image.fromarray(frame_rgb)
    W, H    = pil_img.size

    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_input  = "watermark, logo, text, subtitle, caption, brand name, signature"

    inputs = processor(
        text=task_prompt + text_input,
        images=pil_img,
        return_tensors="pt",
    ).to(DEVICE, torch.float16 if USE_CUDA else torch.float32)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(W, H),
    )

    raw_boxes = []
    try:
        result = parsed.get(task_prompt, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("bboxes_labels", [""] * len(bboxes))
        scores = result.get("bboxes_scores", [1.0] * len(bboxes))

        CONF_THR = 0.25
        PAD      = 20

        for bbox, label, score in zip(bboxes, labels, scores):
            if score < CONF_THR:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1 - PAD);  y1 = max(0, y1 - PAD)
            x2 = min(W, x2 + PAD);  y2 = min(H, y2 + PAD)
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            raw_boxes.append({
                'x': x1, 'y': y1,
                'w': x2 - x1, 'h': y2 - y1,
                'label': label,
            })
    except Exception as e:
        print(f"[FLORENCE] parse err: {e}", flush=True)

    return raw_boxes


def auto_detect_boxes_florence(video_path: str, width: int, height: int) -> list[dict]:
    """
    Esantioneaza 5 frame-uri, ruleaza Florence-2, mergeaza box-urile.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    samples   = [0.05, 0.20, 0.40, 0.65, 0.90]
    all_boxes = []

    for s in samples:
        idx = max(0, min(int(total * s), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = detect_with_florence(frame_rgb)
        print(f"[DETECT] @{s:.0%}: {len(boxes)} box(uri)", flush=True)
        for b in boxes:
            print(f"         '{b['label']}' ({b['w']}x{b['h']} @ {b['x']},{b['y']})", flush=True)
        all_boxes.extend(boxes)

    cap.release()

    plain  = [{'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h']} for b in all_boxes]
    merged = _merge_overlapping(plain, width, height, gap=15)
    print(f"[DETECT] {len(all_boxes)} raw → {len(merged)} merged", flush=True)
    return merged


# ─────────────────────────────────────────────────────────────
# ProPainter — video inpainting cu flow optic
# ─────────────────────────────────────────────────────────────
PROPAINTER_DIR = "/app/ProPainter"

def run_propainter(frames_dir: str, masks_dir: str, output_dir: str,
                   width: int, height: int, fps: float) -> None:
    script = os.path.join(PROPAINTER_DIR, "inference_propainter.py")
    cmd = [
        "python", script,
        "--video",           frames_dir,
        "--mask",            masks_dir,
        "--output",          output_dir,
        "--width",           str(width),
        "--height",          str(height),
        "--neighbor_length", "10",
        "--ref_stride",      "10",
        "--subvideo_length", "80",
        "--raft_iter",       "20",
        "--save_frames",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"} if USE_CUDA else os.environ.copy()

    print(f"[PROPAINTER] Start...", flush=True)
    t0   = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=PROPAINTER_DIR)
    dt   = time.time() - t0

    if proc.returncode != 0:
        print(f"[PROPAINTER] STDERR:\n{proc.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"ProPainter failed (rc={proc.returncode})")
    print(f"[PROPAINTER] Done in {dt:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────
# Utilitare
# ─────────────────────────────────────────────────────────────
def _merge_overlapping(boxes, W, H, gap=8):
    if not boxes:
        return []
    rects = [(b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']) for b in boxes]
    used  = [False] * len(rects)
    out   = []
    for i, r in enumerate(rects):
        if used[i]:
            continue
        x1, y1, x2, y2 = r
        used[i]  = True
        changed  = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if used[j]:
                    continue
                rx1, ry1, rx2, ry2 = rects[j]
                if (rx1 <= x2 + gap and rx2 + gap >= x1 and
                        ry1 <= y2 + gap and ry2 + gap >= y1):
                    x1, y1 = min(x1, rx1), min(y1, ry1)
                    x2, y2 = max(x2, rx2), max(y2, ry2)
                    used[j]  = True
                    changed  = True
        out.append({
            'x': max(0, x1), 'y': max(0, y1),
            'w': min(W, x2) - max(0, x1),
            'h': min(H, y2) - max(0, y1),
        })
    return out


def build_mask_for_boxes(boxes, W, H, feather=8):
    mask   = np.zeros((H, W), dtype=np.uint8)
    DILATE = 8
    for b in boxes:
        x1 = max(0, int(b['x']) - DILATE)
        y1 = max(0, int(b['y']) - DILATE)
        x2 = min(W, int(b['x']) + int(b['w']) + DILATE)
        y2 = min(H, int(b['y']) + int(b['h']) + DILATE)
        mask[y1:y2, x1:x2] = 255
    if feather > 0:
        k    = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask[mask > 0] = 255
    return mask


def _detect_nvenc():
    if not USE_CUDA:
        return False
    try:
        r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                           capture_output=True, text=True, timeout=5)
        return 'h264_nvenc' in r.stdout
    except Exception:
        return False

HAS_NVENC = _detect_nvenc()
print(f"[INIT] Encoder: {'h264_nvenc (GPU)' if HAS_NVENC else 'libx264 (CPU)'}", flush=True)


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────
def process_video(input_path: str, boxes: list, width: int, height: int, fps: float) -> str:
    # Metadata reala
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate',
         '-of', 'json', input_path],
        capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe: {probe.stderr[:200]}")
    meta       = json.loads(probe.stdout)['streams'][0]
    actual_w   = int(meta['width'])
    actual_h   = int(meta['height'])
    num, den   = map(int, meta['r_frame_rate'].split('/'))
    actual_fps = num / den
    print(f"[PROC] Video: {actual_w}x{actual_h} @ {actual_fps:.3f}fps", flush=True)

    # Detectie automata daca nu s-au dat boxes manuale
    if not boxes:
        print("[PROC] Detectez text/logo cu Florence-2...", flush=True)
        boxes = auto_detect_boxes_florence(input_path, actual_w, actual_h)
        if not boxes:
            raise RuntimeError("Florence-2: niciun text/logo detectat. Furnizeaza boxes manual.")
        print(f"[PROC] {len(boxes)} zone de sters", flush=True)

    # Masca
    mask_np = build_mask_for_boxes(boxes, actual_w, actual_h, feather=8)

    # Directoare temporare
    work_dir   = tempfile.mkdtemp(prefix="propainter_")
    frames_dir = os.path.join(work_dir, "frames")
    masks_dir  = os.path.join(work_dir, "masks")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(frames_dir); os.makedirs(masks_dir); os.makedirs(output_dir)

    # Extrage frame-uri
    print("[PROC] Extrag frame-uri...", flush=True)
    t_ext = time.time()
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', input_path,
        '-vf', f'scale={actual_w}:{actual_h}',
        '-start_number', '0',
        os.path.join(frames_dir, '%05d.png'),
    ], check=True, capture_output=True)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.png'))
    n_frames    = len(frame_files)
    print(f"[PROC] {n_frames} frame-uri in {time.time()-t_ext:.1f}s", flush=True)

    # Scrie mastile
    mask_pil = Image.fromarray(mask_np).convert("L")
    for ff in frame_files:
        mask_pil.save(os.path.join(masks_dir, ff))

    # ProPainter
    run_propainter(frames_dir, masks_dir, output_dir, actual_w, actual_h, actual_fps)

    # Gaseste frame-urile output
    pp_frames_dir = os.path.join(output_dir, "frames")
    if not os.path.isdir(pp_frames_dir):
        for root, dirs, files in os.walk(output_dir):
            if any(f.endswith('.png') for f in files):
                pp_frames_dir = root
                break

    out_frames = sorted(f for f in os.listdir(pp_frames_dir) if f.endswith('.png'))
    print(f"[PROC] Frame-uri output: {len(out_frames)}", flush=True)

    # Recompune video cu audio
    final_path = input_path + "_final.mp4"
    if HAS_NVENC:
        vcodec = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'hq',
                  '-rc', 'vbr', '-cq', '23', '-b:v', '0']
    else:
        vcodec = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']

    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(actual_fps),
        '-i', os.path.join(pp_frames_dir, '%05d.png'),
        '-i', input_path,
        '-map', '0:v:0', '-map', '1:a?',
        *vcodec,
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy', '-movflags', '+faststart',
        final_path,
    ], check=True, capture_output=True)

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"[PROC] Video final: {size_mb:.1f} MB", flush=True)

    shutil.rmtree(work_dir, ignore_errors=True)
    return final_path


# ─────────────────────────────────────────────────────────────
# RunPod handler
# ─────────────────────────────────────────────────────────────
def handler(job):
    inp      = job.get('input', {})
    boxes    = inp.get('boxes', [])
    width    = int(inp.get('width', 0))
    height   = int(inp.get('height', 0))
    fps      = float(inp.get('fps', 30.0))
    callback = inp.get('callback_url', '')
    job_id   = inp.get('job_id', job.get('id', 'unknown'))

    tmp        = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    input_path = tmp.name
    tmp.close()

    try:
        if 'video_url' in inp:
            print(f"[DL] {inp['video_url']}", flush=True)
            r = requests.get(inp['video_url'], timeout=300, stream=True)
            r.raise_for_status()
            with open(input_path, 'wb') as f:
                for chunk in r.iter_content(8 * 1024 * 1024):
                    f.write(chunk)
            print(f"[DL] {os.path.getsize(input_path)/1024/1024:.1f} MB", flush=True)
        elif 'video_base64' in inp:
            with open(input_path, 'wb') as f:
                f.write(base64.b64decode(inp['video_base64']))
        else:
            return {'error': 'Niciun video furnizat (video_url sau video_base64)'}

        out     = process_video(input_path, boxes, width, height, fps)
        size_mb = os.path.getsize(out) / 1024 / 1024

        if callback:
            print(f"[UPLOAD] POST → {callback}", flush=True)
            with open(out, 'rb') as f:
                resp = requests.post(
                    callback,
                    files={'video': ('result.mp4', f, 'video/mp4')},
                    data={'job_id': job_id},
                    timeout=300,
                )
            if resp.ok:
                print(f"[UPLOAD] OK {resp.status_code}", flush=True)
                return {'result_uploaded': True, 'job_id': job_id, 'size_mb': round(size_mb, 1)}
            print(f"[UPLOAD] FAILED {resp.status_code} — fallback base64", flush=True)

        with open(out, 'rb') as f:
            return {'video_base64': base64.b64encode(f.read()).decode()}

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        return {'error': str(e)}

    finally:
        for p in [input_path, input_path + '_final.mp4']:
            try:
                os.remove(p)
            except Exception:
                pass


print("[INIT] Worker pornit.", flush=True)
runpod.serverless.start({'handler': handler})