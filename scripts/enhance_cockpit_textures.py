from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import numpy as np
import subprocess
import torch

ROOT = Path(".")
WORK = Path("texture-work")
SRC = WORK / "source"
PNG = WORK / "png"
AI = WORK / "ai"
OUT = WORK / "out"
for folder in (SRC, PNG, AI, OUT):
    folder.mkdir(parents=True, exist_ok=True)

texconv = Path("tools/texconv.exe")
model_path = Path("tools/RealESRGAN_x4plus.pth")
if not texconv.exists():
    raise SystemExit("tools/texconv.exe is missing")

# The first upload commit is the untouched source set. Always upscale from
# those originals, rather than repeatedly enlarging already-upscaled files.
ORIGINAL_COMMITS = ("e672b443031f9fe1dbe5edb22b06bc3915857b1c", "ef37c607fc53e441a6fc5f15f6c942f1e5f7e877", "9897decd68ddef2089629beb1f9106e18c3ac584")
dds_files = sorted(ROOT.glob("int*.dds"))
if not dds_files:
    raise SystemExit("No int*.dds cockpit textures found")

def run(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, check=True)

for dds in dds_files:
    source = SRC / dds.name
    original = None
    for commit in ORIGINAL_COMMITS:
        result = subprocess.run(
            ["git", "show", f"{commit}:{dds.name}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0 and result.stdout:
            original = result.stdout
            print(f"Using original {dds.name} from {commit[:7]}", flush=True)
            break
    if original is not None:
        source.write_bytes(original)
    else:
        source.write_bytes(dds.read_bytes())
        print(f"WARNING: original blob unavailable; using current {dds.name}", flush=True)

    run([str(texconv), "-y", "-ft", "png", "-o", str(PNG), str(source)])
    png = PNG / (dds.stem + ".png")
    if not png.exists():
        raise SystemExit(f"texconv did not produce {png}")

    with Image.open(png) as im:
        im.load()
        rgba = im.convert("RGBA")
    width, height = rgba.size
    longest = max(width, height)
    target_size = (max(1, round(width * 4096 / longest)),
                   max(1, round(height * 4096 / longest)))

    # Real-ESRGAN reconstructs edges and texture detail; Lanczos alone merely
    # interpolates existing pixels and was the source of the soft/pixelated look.
    rgb_input = PNG / (dds.stem + "_rgb.png")
    ai_output = AI / (dds.stem + "_ai.png")
    rgba.convert("RGB").save(rgb_input, format="PNG")
    if not model_path.exists():
        raise SystemExit(f"CPU Real-ESRGAN model is missing: {model_path}")
    # PyTorch CPU inference works on GitHub-hosted runners without Vulkan/GPU.
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path=str(model_path), model=model,
        tile=256, tile_pad=10, pre_pad=0, half=False,
        device=torch.device("cpu"))
    bgr = cv2.imread(str(rgb_input), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read RGB input image {rgb_input}")
    print(f"Running CPU Real-ESRGAN for {dds.name}; this may take several minutes.", flush=True)
    enhanced_bgr, _ = upsampler.enhance(bgr, outscale=4)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    restored_rgb = Image.fromarray(enhanced_rgb).resize(target_size, Image.Resampling.LANCZOS)

    restored = ImageEnhance.Contrast(restored_rgb).enhance(1.025)
    restored = restored.filter(ImageFilter.UnsharpMask(radius=0.7, percent=18, threshold=5))
    alpha = rgba.getchannel("A").resize(target_size, Image.Resampling.LANCZOS)
    restored.putalpha(alpha)
    processed = PNG / (dds.stem + "_enhanced.png")
    restored.save(processed, format="PNG")

    run([str(texconv), "-y", "-m", "0", "-f", "BC3_UNORM", "-o", str(OUT), str(processed)])
    generated = OUT / (dds.stem + "_enhanced.dds")
    if not generated.exists():
        raise SystemExit(f"texconv did not produce {generated}")
    dds.write_bytes(generated.read_bytes())
    with Image.open(processed) as verify:
        print(f"Done {dds.name}: {width}x{height} -> {verify.width}x{verify.height}; DDS {dds.stat().st_size} bytes", flush=True)

print(f"Successfully CPU AI-enhanced and upscaled {len(dds_files)} cockpit DDS textures.")
