from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import subprocess
import sys

ROOT = Path(".")
WORK = Path("texture-work")
SRC = WORK / "source"
PNG = WORK / "png"
OUT = WORK / "out"
for p in (SRC, PNG, OUT):
    p.mkdir(parents=True, exist_ok=True)

texconv = Path("tools/texconv.exe")
dds_files = sorted(ROOT.glob("int*.dds"))
if not dds_files:
    raise SystemExit("No int*.dds cockpit textures found")

def run(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, check=True)

for dds in dds_files:
    # Keep an untouched source copy for this processing pass.
    source = SRC / dds.name
    source.write_bytes(dds.read_bytes())
    run([str(texconv), "-y", "-ft", "png", "-o", str(PNG), str(source)])
    png = PNG / (dds.stem + ".png")
    if not png.exists():
        raise SystemExit(f"texconv did not produce {png}")

    with Image.open(png) as im:
        im.load()
        # Upscale to 4K on the longest edge, preserving aspect ratio.
        # Lanczos provides a high-quality resampling baseline; the sharpening
        # pass below restores some edge definition after enlargement.
        enhanced = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
        width, height = enhanced.size
        longest = max(width, height)
        if longest < 4096:
            new_size = (
                max(1, round(width * 4096 / longest)),
                max(1, round(height * 4096 / longest)),
            )
            print(f"Upscaling {dds.name}: {width}x{height} -> {new_size[0]}x{new_size[1]}", flush=True)
            enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)
        else:
            print(f"{dds.name} already meets/exceeds 4K: {width}x{height}", flush=True)
        alpha = enhanced.getchannel("A") if enhanced.mode == "RGBA" else None
        rgb = enhanced.convert("RGB")
        rgb = ImageEnhance.Contrast(rgb).enhance(1.06)
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.22)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=45, threshold=4))
        if alpha is not None:
            rgb.putalpha(alpha)
        processed = PNG / (dds.stem + "_enhanced.png")
        rgb.save(processed, format="PNG")

    run([str(texconv), "-y", "-m", "0", "-f", "BC3_UNORM", "-o", str(OUT), str(PNG / (dds.stem + "_enhanced.png"))])
    generated = OUT / (dds.stem + "_enhanced.dds")
    if not generated.exists():
        raise SystemExit(f"texconv did not produce {generated}")
    # Replace source only after a successful round-trip conversion.
    dds.write_bytes(generated.read_bytes())
    print(f"Enhanced/upscaled {dds.name}: {dds.stat().st_size} bytes; output dimensions {Image.open(PNG / (dds.stem + \"_enhanced.png\")).size}", flush=True)

print(f"Successfully processed {len(dds_files)} DDS textures.")
