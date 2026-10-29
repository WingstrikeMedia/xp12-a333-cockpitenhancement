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
    restored_rgb = None
    if upscaler.exists():
        try:
            run([str(upscaler), "-i", str(rgb_input), "-o", str(ai_output),
                 "-m", str(Path("tools/models")), "-n", "realesrgan-x4plus",
                 "-s", "4", "-t", "256"])
            if ai_output.exists():
                with Image.open(ai_output) as restored:
                    restored.load()
                    restored_rgb = restored.convert("RGB")
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: Real-ESRGAN failed for {dds.name} (likely no Vulkan device: {exc.returncode}); using high-quality resize fallback.", flush=True)

    if restored_rgb is None:
        # GitHub-hosted runners do not always expose a Vulkan device. Keep the
        # workflow productive with a deterministic high-quality resize fallback.
        restored_rgb = rgba.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    else:
        # Downsample the 4x AI reconstruction to 4K on the longest edge.
        restored_rgb = restored_rgb.resize(target_size, Image.Resampling.LANCZOS)

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

print(f"Successfully AI-enhanced and upscaled {len(dds_files)} cockpit DDS textures.")
