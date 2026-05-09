"""
Patch vendor VGGT files for Apple Silicon MPS compatibility.

Run `vggt patch` after cloning to apply required patches.
"""

from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[3] / "vendor" / "vggt"


def _patch_utils_file() -> bool:
    """Patch make_sincos_pos_embed to use input dtype instead of hardcoded float32/double"""
    filepath = VENDOR_DIR / "vggt" / "heads" / "utils.py"
    with open(filepath) as f:
        content = f.read()

    if "dtype=pos.dtype" in content:
        print("  ✓ utils.py already patched")
        return False

    old1 = 'dtype=torch.float32 if device.type == "mps" else torch.double'
    new1 = "dtype=pos.dtype"
    content = content.replace(old1, new1)

    old2 = "return emb.float()"
    new2 = "return emb.to(dtype=pos.dtype)"
    content = content.replace(old2, new2)

    with open(filepath, "w") as f:
        f.write(content)
    print("  ✓ utils.py patched")
    return True


def _patch_vggt_file() -> bool:
    """Patch autocast to be device-agnostic"""
    filepath = VENDOR_DIR / "vggt" / "models" / "vggt.py"
    with open(filepath) as f:
        content = f.read()

    if "device_type=images.device.type" in content:
        print("  ✓ vggt.py already patched")
        return False

    old = "with torch.cuda.amp.autocast(enabled=False):"
    new = "with torch.amp.autocast(device_type=images.device.type, enabled=False):"
    content = content.replace(old, new)

    with open(filepath, "w") as f:
        f.write(content)
    print("  ✓ vggt.py patched")
    return True


def apply_vendor_patches():
    """Apply all required vendor patches for MPS compatibility"""
    print("🔧 Patching vendor VGGT for Apple Silicon MPS...")
    patched_any = False
    patched_any |= _patch_utils_file()
    patched_any |= _patch_vggt_file()
    if not patched_any:
        print("   All patches already applied.")
    else:
        print("✅ Vendor patches applied successfully.")
