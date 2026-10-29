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
        # Conservative clarity pass: modest local detail and tonal separation,
        # without recoloring the cockpit or changing its dimensions.
        enhanced = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
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
    print(f"Enhanced {dds.name}: {source.stat().st_size} -> {dds.stat().st_size} bytes", flush=True)

print(f"Successfully processed {len(dds_files)} DDS textures.")
