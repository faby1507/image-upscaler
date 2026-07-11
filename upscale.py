#!/usr/bin/env python3
"""
upscale.py — Image upscaler con Real-ESRGAN + Metal (MPS) per Apple Silicon
Requisiti: pip install torch torchvision pillow basicsr realesrgan

Uso:
  python upscale.py immagine.jpg
  python upscale.py immagine.jpg -s 4 -o output.png
  python upscale.py immagine.jpg -m anime
  python upscale.py cartella/ -o cartella_upscaled/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# ── Dipendenze ──────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg in ("torch", "PIL", "basicsr", "realesrgan"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("❌ Pacchetti mancanti:", ", ".join(missing))
        print("\nInstalla con:\n  pip install torch torchvision pillow basicsr realesrgan\n")
        sys.exit(1)

check_deps()

import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ── Device ──────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon rilevato — uso Metal (MPS)")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA disponibile")
        return "cuda"
    else:
        print("⚠️  Nessuna GPU disponibile — uso CPU (più lento)")
        return "cpu"

# ── Modelli ─────────────────────────────────────────────────────────────────

MODELS = {
    "photo": {
        2: {
            "name": "RealESRGAN_x2plus",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "num_block": 23,
        },
        4: {
            "name": "RealESRGAN_x4plus",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "num_block": 23,
        },
    },
    "anime": {
        2: {
            "name": "RealESRGAN_x2plus",  # anime ha solo x4, usiamo photo x2 come fallback
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "num_block": 23,
        },
        4: {
            "name": "RealESRGAN_x4plus_anime_6B",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "num_block": 6,
        },
    },
}

def load_upsampler(scale: int, mode: str, device: str, model_dir: Path) -> RealESRGANer:
    cfg = MODELS[mode].get(scale)
    if cfg is None:
        print(f"❌ Scale {scale}x non supportato. Usa 2 o 4.")
        sys.exit(1)

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=cfg["num_block"],
        num_grow_ch=32, scale=scale
    )

    model_path = model_dir / f"{cfg['name']}.pth"
    half = device == "cuda"

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(model_path) if model_path.exists() else cfg["url"],
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )
    return upsampler

# ── Processing ───────────────────────────────────────────────────────────────

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def upscale_image(
    src: Path,
    dst: Path,
    upsampler: RealESRGANer,
    scale: int,
) -> None:
    img = Image.open(src).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)

    t0 = time.time()
    output, _ = upsampler.enhance(img_np, outscale=scale)
    elapsed = time.time() - t0

    result = Image.fromarray(output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    result.save(dst)

    w_in, h_in = img.size
    w_out, h_out = result.size
    print(f"  ✓ {src.name} [{w_in}×{h_in} → {w_out}×{h_out}] ({elapsed:.1f}s)")

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Upscaling immagini con Real-ESRGAN + Metal (MPS)"
    )
    p.add_argument("input", help="File immagine o cartella")
    p.add_argument(
        "-s", "--scale", type=int, default=4, choices=[2, 4],
        help="Fattore di ingrandimento: 2 o 4 (default: 4)"
    )
    p.add_argument(
        "-m", "--mode", default="photo", choices=["photo", "anime"],
        help="Modalità: photo (default) o anime"
    )
    p.add_argument(
        "-o", "--output",
        help="File o cartella di output (default: <nome>_upscaled.<ext>)"
    )
    p.add_argument(
        "--model-dir", default="~/.cache/realesrgan",
        help="Cartella dove scaricare/cercare i modelli (default: ~/.cache/realesrgan)"
    )
    return p.parse_args()

def resolve_output(src: Path, dst_arg: Optional[str], is_dir: bool) -> Path:
    if dst_arg:
        return Path(dst_arg).expanduser()
    if is_dir:
        return src.parent / f"{src.name}_upscaled"
    return src.parent / f"{src.stem}_upscaled{src.suffix}"

def main():
    args = parse_args()
    src = Path(args.input).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"❌ Percorso non trovato: {src}")
        sys.exit(1)

    device = get_device()
    label = "anime 🎌" if args.mode == "anime" else "photo 📷"
    print(f"🔍 Carico modello Real-ESRGAN x{args.scale} [{label}]...\n")
    upsampler = load_upsampler(args.scale, args.mode, device, model_dir)

    if src.is_file():
        if src.suffix.lower() not in SUPPORTED:
            print(f"❌ Formato non supportato: {src.suffix}")
            sys.exit(1)
        dst = resolve_output(src, args.output, is_dir=False)
        upscale_image(src, dst, upsampler, args.scale)
        print(f"\n💾 Salvato: {dst}")

    elif src.is_dir():
        files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in SUPPORTED]
        if not files:
            print("❌ Nessuna immagine trovata nella cartella.")
            sys.exit(1)

        dst_dir = resolve_output(src, args.output, is_dir=True)
        print(f"📁 {len(files)} immagini trovate → {dst_dir}\n")

        for f in files:
            out = dst_dir / f.name
            upscale_image(f, out, upsampler, args.scale)

        print(f"\n✅ Fatto! {len(files)} immagini salvate in: {dst_dir}")

if __name__ == "__main__":
    main()