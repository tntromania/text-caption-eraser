#!/usr/bin/env python3
"""
RunPod worker — text/logo removal din video
  Detectie  : Florence-2 (microsoft/Florence-2-large)
  Inpainting: ProPainter  (video-aware, no flickering)
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
# ─────────────────────────────────────────────────────────────
_FLORENCE_MODEL     = None
_FLORENCE_PROCESSOR = None

def get_florence():
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    if _FLORENCE_MODEL is None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        MODEL_ID = "microsoft/Florence-2-large"
        print(f"[INIT] Incarc Florence-2 ({MODEL_ID})...", flush=True)
        _FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        _FLORENCE_PROCESSOR.image_processor.do_resize = False  # evita resize intern

        _FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if USE_CUDA else torch.float32,
            trust_remote_code=True,
        ).to(DEVICE)
        _FLORENCE_MODEL.eval()
        print("[INIT] Florence-2 ready!", flush=True)
    return _FLORENCE_MODEL, _FLORENCE_PROCESSOR


def detect_with_florence(frame_rgb: np.ndarray, prompt: str = "<OD>") -> list[dict]:
    """
    Ruleaza Florence-2 pe un frame RGB si returneaza boxes:
    [{'x': int, 'y': int, 'w': int, 'h': int, 'label': str}, ...]

    Prompt-uri utile:
      "<OD>"            → Object Detection generic
      "<CAPTION_TO_PHRASE_GROUNDING>" cu text_input
                        → grounding dupa descriere
      "<DENSE_REGION_CAPTION>" → caption dens cu bbox

    Folosim "<OD>" + filtrare pe label pentru text/logo.
    """
    model, processor = get_florence()
    pil_img = Image.fromarray(frame_rgb)
    W, H    = pil_img.size

    # GroundedOD cu prompt explicit pentru ce vrem sa eliminam
    task_prompt  = "<OPEN_VOCABULARY_DETECTION>"
    text_input   = "watermark, logo, text, subtitle, caption, brand name, signature"

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
        # parsed e dict cu cheia task_prompt
        result = parsed.get(task_prompt, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("bboxes_labels", [""] * len(bboxes))
        scores = result.get("bboxes_scores", [1.0] * len(bboxes))

        CONF_THR = 0.25
        PAD      = 20   # padding px in jurul detectiei

        for bbox, label, score in zip(bboxes, labels, scores):
            if score < CONF_THR:
                continue
            # Florence returneaza [x1,y1,x2,y2] in coordonate absolute
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
    Eșantionează 5 frame-uri din video, rulează Florence-2,
    mergeaza box-urile suprapuse din toate frame-urile.
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
        print(f"[DETECT] @{s:.0%}: {len(boxes)} box(uri) detectate", flush=True)
        for b in boxes:
            print(f"         '{b['label']}' ({b['w']}x{b['h']} @ {b['x']},{b['y']})", flush=True)
        all_boxes.extend(boxes)

    cap.release()

    # Scoatem 'label' inainte de merge (merge lucreaza cu x/y/w/h)
    plain = [{'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h']} for b in all_boxes]
    merged = _merge_overlapping(plain, width, height, gap=15)
    print(f"[DETECT] {len(all_boxes)} raw → {len(merged)} merged", flush=True)
    return merged


# ─────────────────────────────────────────────────────────────
# ProPainter — video inpainting cu flow optic
# ─────────────────────────────────────────────────────────────
PROPAINTER_DIR = "/app/ProPainter"

def _ensure_propainter():
    """Cloneaza ProPainter + descarca weights daca lipsesc."""
    if not os.path.isdir(PROPAINTER_DIR):
        print("[PROPAINTER] Clonez repo...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/sczhou/ProPainter.git",
             PROPAINTER_DIR],
            check=True, capture_output=True
        )
        # Instaleaza dependinte ProPainter
        subprocess.run(
            ["pip", "install", "--no-cache-dir", "-r",
             os.path.join(PROPAINTER_DIR, "requirements.txt")],
            check=True, capture_output=True
        )
        print("[PROPAINTER] Repo + deps OK", flush=True)

    # Weights
    weights_dir = os.path.join(PROPAINTER_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    needed = {
        "ProPainter.pth":
            "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        "recurrent_flow_completion.pth":
            "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        "raft-things.pth":
            "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    }
    for fname, url in needed.items():
        dst = os.path.join(weights_dir, fname)
        if not os.path.isfile(dst):
            print(f"[PROPAINTER] Download {fname}...", flush=True)
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(8 * 1024 * 1024):
                    f.write(chunk)
            print(f"[PROPAINTER] {fname} OK ({os.path.getsize(dst)//1024//1024} MB)", flush=True)


def run_propainter(frames_dir: str, masks_dir: str, output_dir: str,
                   width: int, height: int, fps: float,
                   neighbor_length: int = 10, ref_stride: int = 10) -> None:
    """
    Apeleaza ProPainter CLI pe un set de frame-uri + masti.
    frames_dir : PNG-uri input  (00000.png, 00001.png, ...)
    masks_dir  : PNG-uri masca  (00000.png, ...) — alb=zona de sters
    output_dir : unde scrie ProPainter rezultatele
    """
    _ensure_propainter()

    script = os.path.join(PROPAINTER_DIR, "inference_propainter.py")
    cmd = [
        "python", script,
        "--video",      frames_dir,
        "--mask",       masks_dir,
        "--output",     output_dir,
        "--width",      str(width),
        "--height",     str(height),
        "--neighbor_length", str(neighbor_length),
        "--ref_stride",      str(ref_stride),
        "--subvideo_length", "80",   # proceseaza in ferestre de 80 frame-uri
        "--raft_iter",       "20",
        "--save_frames",              # vrem frame-uri individuale, nu direct video
    ]
    if USE_CUDA:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    else:
        env = os.environ.copy()

    print(f"[PROPAINTER] Rulez: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=PROPAINTER_DIR)
    dt   = time.time() - t0

    if proc.returncode != 0:
        print(f"[PROPAINTER] STDERR:\n{proc.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"ProPainter failed (rc={proc.returncode})")
    print(f"[PROPAINTER] Done in {dt:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────
# Utilitare comune
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
    """Masca binara (0/255) cu optional gaussian feather."""
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
        r = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        return 'h264_nvenc' in r.stdout
    except Exception:
        return False

HAS_NVENC = _detect_nvenc()
print(f"[INIT] Encoder: {'h264_nvenc (GPU)' if HAS_NVENC else 'libx264 (CPU)'}", flush=True)


# ─────────────────────────────────────────────────────────────
# Pipeline principal — ProPainter pe batch de frame-uri
# ─────────────────────────────────────────────────────────────
def process_video(input_path: str, boxes: list, width: int, height: int, fps: float) -> str:
    # 1. Metadata video reala
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

    # 2. Detectie automata daca nu s-au dat boxes manuale
    if not boxes:
        print("[PROC] Detectez text/logo cu Florence-2...", flush=True)
        boxes = auto_detect_boxes_florence(input_path, actual_w, actual_h)
        if not boxes:
            raise RuntimeError("Florence-2: niciun text/logo detectat. Furnizeaza boxes manual.")
        print(f"[PROC] {len(boxes)} zone de sters", flush=True)

    # 3. Construieste masca (aceeasi pentru toate frame-urile — static mask)
    mask_np = build_mask_for_boxes(boxes, actual_w, actual_h, feather=8)

    # 4. Extrage toate frame-urile video in directoare temporare
    work_dir    = tempfile.mkdtemp(prefix="propainter_")
    frames_dir  = os.path.join(work_dir, "frames")
    masks_dir   = os.path.join(work_dir, "masks")
    output_dir  = os.path.join(work_dir, "output")
    os.makedirs(frames_dir);  os.makedirs(masks_dir);  os.makedirs(output_dir)

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
    print(f"[PROC] {n_frames} frame-uri extrase in {time.time()-t_ext:.1f}s", flush=True)

    # 5. Salveaza masca pentru fiecare frame
    mask_pil = Image.fromarray(mask_np).convert("L")
    for ff in frame_files:
        mask_pil.save(os.path.join(masks_dir, ff))
    print(f"[PROC] Masti scrise ({n_frames}x)", flush=True)

    # 6. Ruleaza ProPainter
    print("[PROC] Pornesc ProPainter...", flush=True)
    t_pp = time.time()
    run_propainter(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        width=actual_w,
        height=actual_h,
        fps=actual_fps,
        neighbor_length=10,
        ref_stride=10,
    )
    print(f"[PROC] ProPainter done in {time.time()-t_pp:.1f}s", flush=True)

    # 7. ProPainter scrie frame-urile intr-un subfolder "frames"
    pp_frames_dir = os.path.join(output_dir, "frames")
    if not os.path.isdir(pp_frames_dir):
        # fallback: cauta recursiv primul dir cu PNG-uri
        for root, dirs, files in os.walk(output_dir):
            if any(f.endswith('.png') for f in files):
                pp_frames_dir = root
                break

    out_frames = sorted(f for f in os.listdir(pp_frames_dir) if f.endswith('.png'))
    print(f"[PROC] Frame-uri output: {len(out_frames)}", flush=True)

    # 8. Recompune video cu audio original
    final_path = input_path + "_final.mp4"

    if HAS_NVENC:
        vcodec_args = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '23', '-b:v', '0']
    else:
        vcodec_args = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']

    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(actual_fps),
        '-i', os.path.join(pp_frames_dir, '%05d.png'),
        '-i', input_path,
        '-map', '0:v:0', '-map', '1:a?',
        *vcodec_args,
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy', '-movflags', '+faststart',
        final_path,
    ], check=True, capture_output=True)

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"[PROC] Video final: {size_mb:.1f} MB → {final_path}", flush=True)

    # 9. Curata temporarele
    shutil.rmtree(work_dir, ignore_errors=True)

    return final_path


# ─────────────────────────────────────────────────────────────
# RunPod handler
# ─────────────────────────────────────────────────────────────
def handler(job):
    inp      = job.get('input', {})
    boxes    = inp.get('boxes', [])          # optional — daca lipseste, detectam auto
    width    = int(inp.get('width', 0))
    height   = int(inp.get('height', 0))
    fps      = float(inp.get('fps', 30.0))
    callback = inp.get('callback_url', '')
    job_id   = inp.get('job_id', job.get('id', 'unknown'))

    tmp        = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    input_path = tmp.name
    tmp.close()

    try:
        # Download sau decode video input
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

        out      = process_video(input_path, boxes, width, height, fps)
        size_mb  = os.path.getsize(out) / 1024 / 1024

        # Upload callback sau base64 fallback
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
