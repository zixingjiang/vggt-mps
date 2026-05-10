"""
Benchmark command for VGGT-MPS performance testing
"""

import sys
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt_mps.config import DEVICE, get_model_path, is_model_available
from vggt_mps.vggt_core import VGGTProcessor


def run_benchmark(args):
    """Run performance benchmarks"""
    print("=" * 60)
    print("⚡ VGGT Performance Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Images: {args.images}")
    print(f"Compare: {args.compare}")
    print("-" * 60)

    # Check model availability
    if not is_model_available():
        print("\n❌ VGGT model not found!")
        print("Run: python main.py download")
        print("\nUsing simulated mode for benchmark...")

    # Create synthetic test images
    print("\n📸 Creating synthetic test images...")
    images = []
    for i in range(args.images):
        # Create random image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        images.append(img_array)

    # Initialize processor
    processor = VGGTProcessor(device=DEVICE, precision='fp16' if getattr(args, 'half', False) else 'fp32')

    results = {}

    # Benchmark regular VGGT
    print("\n🔵 Benchmarking Regular VGGT...")

    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if DEVICE.type == "cuda" else 0

    try:
        regular_output = processor.process_images(images)
        regular_time = time.time() - start_time
        regular_memory = torch.cuda.memory_allocated() if DEVICE.type == "cuda" else 0
        regular_memory_used = (regular_memory - start_memory) / 1024 / 1024  # MB

        results['regular'] = {
            'success': True,
            'time': regular_time,
            'memory': regular_memory_used,
            'fps': args.images / regular_time
        }
        print(f"  ✅ Time: {regular_time:.2f}s")
        print(f"  ✅ FPS: {args.images / regular_time:.2f}")
        if DEVICE.type == "cuda":
            print(f"  ✅ Memory: {regular_memory_used:.1f} MB")

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results['regular'] = {'success': False, 'error': str(e)}

    # Print summary
    print("\n✅ Benchmark complete!")
    return 0