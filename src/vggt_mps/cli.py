"""
Central CLI entry point for VGGT-MPS.
All command-line argument parsing and dispatch lives here.
Uses tyro for typed argument parsing via dataclasses.
"""

import argparse
import sys
from dataclasses import dataclass, asdict
from typing import Annotated, Literal, Optional, Union

import tyro
import tyro.conf


# ── Subcommand dataclasses ──────────────────────────────────────────────────

@dataclass
class Reconstruct:
    """3D reconstruction from images"""
    images: Annotated[tuple[str, ...], tyro.conf.Positional]
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    output: str = "outputs"
    export: Optional[Literal["ply", "obj", "glb"]] = None


@dataclass
class Gradio:
    """Launch Gradio 3D reconstruction UI"""
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    port: int = 7860
    share: bool = False


@dataclass
class Viser:
    """Launch Viser 3D viewer"""
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    image_folder: str = "examples/kitchen/images/"
    use_point_map: bool = False
    background_mode: bool = False
    port: int = 8080
    conf_threshold: float = 25.0
    mask_sky: bool = False


@dataclass
class Colmap:
    """COLMAP-format 3D reconstruction"""
    scene_dir: str
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    seed: int = 42
    use_ba: bool = False
    max_reproj_error: float = 8.0
    shared_camera: bool = False
    camera_type: str = "SIMPLE_PINHOLE"
    vis_thresh: float = 0.2
    query_frame_num: int = 8
    max_query_pts: int = 4096
    fine_tracking: bool = True
    conf_thres_value: float = 5.0


@dataclass
class Web:
    """Launch web interface"""
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    port: int = 7860
    share: bool = False


@dataclass
class Test:
    """Run tests"""
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    suite: Literal["all", "mps", "sparse", "quick"] = "quick"


@dataclass
class Benchmark:
    """Benchmark performance"""
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    images: int = 10
    compare: bool = False


@dataclass
class Download:
    """Download VGGT model"""
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False
    source: Literal["huggingface", "direct"] = "huggingface"


@dataclass
class Patch:
    """Patch vendor VGGT files for MPS compatibility"""
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False


@dataclass
class Quick:
    """Quick-start reconstruction with example datasets"""
    dataset: Literal[
        "kitchen", "room", "llff_flower", "llff_fern",
        "single_oil_painting", "single_cartoon",
    ] = "kitchen"
    images: int = 4
    precision: Literal["fp32", "fp16"] = "fp32"
    sparse: bool = False


# ── Subcommand groups ─────────────────────────────────────────────────────────

DemoSubcommand = Annotated[
    Union[Quick, Gradio, Viser, Colmap],
    tyro.conf.subcommand(name="demo", description="Run vendor-mirrored demos"),
]

# ── Dispatch ─────────────────────────────────────────────────────────────────

def main() -> None:
    parsed = tyro.cli(
        Union[Reconstruct, DemoSubcommand, Web, Test, Benchmark, Download, Patch],
        description="VGGT 3D Reconstruction on Apple Silicon",
    )

    from vggt_mps.config import set_precision, set_sparse_enabled
    set_precision(parsed.precision)
    set_sparse_enabled(parsed.sparse)

    try:
        if isinstance(parsed, Quick):
            args = argparse.Namespace(**asdict(parsed))
            from .demo.quick import run as run_quick
            run_quick(args)

        elif isinstance(parsed, Reconstruct):
            args = argparse.Namespace(**asdict(parsed))
            from .commands.reconstruct import run_reconstruction
            run_reconstruction(args)

        elif isinstance(parsed, Gradio):
            from .demo.gradio import run as run_gradio
            run_gradio(port=parsed.port, share=parsed.share)

        elif isinstance(parsed, Viser):
            args = argparse.Namespace(**asdict(parsed))
            from .demo.viser import run as run_viser
            run_viser(args)

        elif isinstance(parsed, Colmap):
            args = argparse.Namespace(**asdict(parsed))
            from .demo.colmap import run as run_colmap
            run_colmap(args)

        elif isinstance(parsed, Web):
            args = argparse.Namespace(**asdict(parsed))
            from .commands.web_interface import launch_web_interface
            launch_web_interface(args)

        elif isinstance(parsed, Test):
            args = argparse.Namespace(**asdict(parsed))
            from .commands.test_runner import run_tests
            run_tests(args)

        elif isinstance(parsed, Benchmark):
            args = argparse.Namespace(**asdict(parsed))
            from .commands.benchmark import run_benchmark
            run_benchmark(args)

        elif isinstance(parsed, Download):
            args = argparse.Namespace(**asdict(parsed))
            from .commands.download_model import download_model
            download_model(args)

        elif isinstance(parsed, Patch):
            from .commands.patch_vendor import apply_vendor_patches
            apply_vendor_patches()

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(130)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing dependencies: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 For help, run: vggt --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
