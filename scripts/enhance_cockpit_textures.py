from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import subprocess
import sys
import io

ROOT = Path(".")
WORK = Path("texture-work")
SRC = WORK / "source"
PNG = WORK / "png"
AI = WORK / "ai"
OUT = WORK / "out"
for folder in (SRC, PNG, AI, OUT):
    folder.mkdir(parents=True, exist_ok=True)

texconv = Path("tools/texconv.exe")
upscaler = Path("tools/realesrgan-ncnn-vulkan.exe")
if not texconv.exists():
    raise SystemExit("tools/texconv.exe is missing")
if not upscaler.exists():
    raise SystemExit("tools/realesrgan-ncnn-vulkan.exe is missing")

# The first upload commit is the untouched source set. Always upscale from
# those originals, rather than repeatedly enlarging already-upscaled files.
ORIGINAL_COMMIT = "e672b443031f9fe1dbe5edb22b06bc3915857b1c"
dds_files = sorted(ROOT.glob("int*.dds"))
if not dds_files:
    raise SystemExit("No int*.dds cockpit textures found")

def run(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, check=True)

for dds in dds_files:
    source = SRC / dds.name
    result = subprocess.run(
        ["git", "show", f"{ORIGINAL_COMMIT}:{dds.name}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode == 0 and result.stdout:
        source.write_bytes(result.stdout)
        print(f"Using original repository texture for {dds.name}", flush=True)
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
    run([str(upscaler), "-i", str(rgb_input), "-o", str(ai_output),
         "-n", "realesrgan-x4plus", "-s", "4", "-t", "256"])
    if not ai_output.exists():
        raise SystemExit(f"Real-ESRGAN did not produce {ai_output}")

    with Image.open(ai_output) as restored:
        restored.load()
        restored = restored.convert("RGB")
        # Downsample the 4x reconstruction to 4K on the longest edge. This
        # averages artifacts and keeps texture dimensions practical for XP12.
        restored = restored.resize(target_size, Image.Resampling.LANCZOS)
        restored = ImageEnhance.Contrast(restored).enhance(1.025)
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

print(f"Successfully AI-enhanced and upscaled {len(dds_files)} cockpit DDS textures.")
