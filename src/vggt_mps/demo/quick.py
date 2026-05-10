"""Quick-start reconstruction with vendored example datasets."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vggt_mps.config import (
    DEVICE,
    OUTPUT_DIR,
    TEST_DATA,
    get_precision,
)
from vggt_mps.vggt_core import VGGTProcessor
from vggt_mps.visualization import create_visualizations


def run(args):
    dataset = args.dataset
    num_images = args.images

    _KNOWN_DATASETS = {
        "kitchen", "room", "llff_flower", "llff_fern",
        "single_oil_painting", "single_cartoon",
    }

    if dataset not in _KNOWN_DATASETS:
        print(f"❌ Unknown dataset: {dataset}")
        print(f"   Available: {', '.join(sorted(_KNOWN_DATASETS))}")
        sys.exit(1)

    image_dir = TEST_DATA.get(dataset)

    if not image_dir.exists():
        print(f"❌ Dataset directory not found: {image_dir}")
        print("   Make sure the vendor submodule is initialized: git submodule update --init")
        sys.exit(1)

    image_paths = sorted(image_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if len(image_paths) == 0:
        print(f"❌ No images found in {image_dir}")
        sys.exit(1)

    use_count = min(num_images, len(image_paths))
    image_paths = image_paths[:use_count]

    print("=" * 60)
    print(f"🔮 VGGT Quick-Start: {dataset}")
    print("=" * 60)
    print(f"Device:   {DEVICE}")
    print(f"Dataset:  {dataset} ({len(image_paths)} / {use_count} images)")
    print(f"Precision: {get_precision()}")
    print("-" * 60)

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(np.array(img))
        print(f"  Loaded: {p.name} ({img.size[0]}x{img.size[1]})")

    print(f"\n🔮 Running VGGT reconstruction on {len(images)} images...")
    processor = VGGTProcessor(device=DEVICE, precision=get_precision())

    try:
        result = processor.process_images(images)
    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")
        sys.exit(1)

    if isinstance(result, dict):
        depth_maps = result.get("depth_maps", [])
        camera_poses = result.get("camera_poses")
        point_cloud = result.get("point_cloud")
        point_colors = result.get("point_colors")
    else:
        depth_maps = result
        camera_poses = None
        point_cloud = None
        point_colors = None

    if not depth_maps:
        print("❌ No depth maps produced")
        sys.exit(1)

    print(f"\n📊 Creating visualizations...")
    output_files = create_visualizations(
        images,
        depth_maps,
        OUTPUT_DIR,
        camera_poses=camera_poses,
        point_cloud=point_cloud,
        point_colors=point_colors,
        show=True,
    )

    print("\n" + "=" * 60)
    print("✅ Quick-start complete!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    for f in output_files:
        print(f"   - {f.name}")
    print("=" * 60)
