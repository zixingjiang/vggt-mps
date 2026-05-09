"""
Central CLI entry point for VGGT-MPS.
All command-line argument parsing and dispatch lives here.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT 3D Reconstruction on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16"], default="fp32",
        help="Inference precision (fp32 for full precision, fp16 for faster inference with less memory)",
    )
    parser.add_argument(
        "--sparse", action="store_true",
        help="Use sparse attention (O(n) memory scaling for many images)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── demo {gradio, viser, colmap} ──────────────────────────────────────
    demo_parser = subparsers.add_parser("demo", help="Run vendor-mirrored demos")
    demo_sub = demo_parser.add_subparsers(dest="demo_cmd", help="Demo to run")

    # demo gradio
    gradio_parser = demo_sub.add_parser("gradio", help="Launch Gradio 3D reconstruction UI")
    gradio_parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    gradio_parser.add_argument("--share", action="store_true", help="Create public share link")

    # demo viser (mirrors vendor/vggt/demo_viser.py CLI)
    viser_parser = demo_sub.add_parser("viser", help="Launch Viser 3D viewer")
    viser_parser.add_argument(
        "--image_folder", type=str, default="examples/kitchen/images/",
        help="Path to folder containing images",
    )
    viser_parser.add_argument(
        "--use_point_map", action="store_true",
        help="Use point map instead of depth-based points",
    )
    viser_parser.add_argument(
        "--background_mode", action="store_true",
        help="Run the viser server in background mode",
    )
    viser_parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    viser_parser.add_argument(
        "--conf_threshold", type=float, default=25.0,
        help="Initial percentage of low-confidence points to filter out",
    )
    viser_parser.add_argument(
        "--mask_sky", action="store_true",
        help="Apply sky segmentation to filter out sky points",
    )

    # demo colmap (mirrors vendor/vggt/demo_colmap.py CLI)
    colmap_parser = demo_sub.add_parser("colmap", help="COLMAP-format 3D reconstruction")
    colmap_parser.add_argument(
        "--scene_dir", type=str, required=True,
        help="Directory containing the scene images",
    )
    colmap_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    colmap_parser.add_argument(
        "--use_ba", action="store_true", default=False,
        help="Use BA for reconstruction",
    )
    colmap_parser.add_argument(
        "--max_reproj_error", type=float, default=8.0,
        help="Maximum reprojection error for reconstruction",
    )
    colmap_parser.add_argument(
        "--shared_camera", action="store_true", default=False,
        help="Use shared camera for all images",
    )
    colmap_parser.add_argument(
        "--camera_type", type=str, default="SIMPLE_PINHOLE",
        help="Camera type for reconstruction",
    )
    colmap_parser.add_argument(
        "--vis_thresh", type=float, default=0.2,
        help="Visibility threshold for tracks",
    )
    colmap_parser.add_argument(
        "--query_frame_num", type=int, default=8,
        help="Number of frames to query",
    )
    colmap_parser.add_argument(
        "--max_query_pts", type=int, default=4096,
        help="Maximum number of query points",
    )
    colmap_parser.add_argument(
        "--fine_tracking", action="store_true", default=True,
        help="Use fine tracking (slower but more accurate)",
    )
    colmap_parser.add_argument(
        "--conf_thres_value", type=float, default=5.0,
        help="Confidence threshold value for depth filtering (wo BA)",
    )

    # ── Existing commands ─────────────────────────────────────────────────
    recon_parser = subparsers.add_parser("reconstruct", help="3D reconstruction from images")
    recon_parser.add_argument("images", nargs="+", help="Image files to process")
    recon_parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    recon_parser.add_argument("--export", choices=["ply", "obj", "glb"], help="Export format")

    web_parser = subparsers.add_parser("web", help="Launch web interface")
    web_parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    web_parser.add_argument("--share", action="store_true", help="Create public link")

    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--suite", choices=["all", "mps", "sparse", "quick"],
        default="quick", help="Test suite to run",
    )

    bench_parser = subparsers.add_parser("benchmark", help="Benchmark performance")
    bench_parser.add_argument("--images", type=int, default=10, help="Number of images")
    bench_parser.add_argument("--compare", action="store_true", help="Compare sparse vs dense")

    download_parser = subparsers.add_parser("download", help="Download VGGT model")
    download_parser.add_argument(
        "--source", choices=["huggingface", "direct"],
        default="huggingface", help="Download source",
    )

    patch_parser = subparsers.add_parser("patch", help="Patch vendor VGGT files for MPS compatibility")

    args = parser.parse_args()

    from vggt_mps.config import set_precision, set_sparse_enabled
    set_precision(args.precision)
    set_sparse_enabled(args.sparse)

    if not args.command:
        parser.print_help()
        return

    try:
        # ── Demo subcommands ──────────────────────────────────────────────
        if args.command == "demo":
            if args.demo_cmd == "gradio":
                from .demo.gradio import run as run_gradio
                run_gradio(port=args.port, share=args.share)

            elif args.demo_cmd == "viser":
                from .demo.viser import run as run_viser
                run_viser(args)

            elif args.demo_cmd == "colmap":
                from .demo.colmap import run as run_colmap
                run_colmap(args)

            else:
                demo_parser.print_help()

        # ── Existing commands ──────────────────────────────────────────────
        elif args.command == "reconstruct":
            from .commands.reconstruct import run_reconstruction
            run_reconstruction(args)

        elif args.command == "web":
            from .commands.web_interface import launch_web_interface
            launch_web_interface(args)

        elif args.command == "test":
            from .commands.test_runner import run_tests
            run_tests(args)

        elif args.command == "benchmark":
            from .commands.benchmark import run_benchmark
            run_benchmark(args)

        elif args.command == "download":
            from .commands.download_model import download_model
            download_model(args)

        elif args.command == "patch":
            from .commands.patch_vendor import apply_vendor_patches
            apply_vendor_patches()

        else:
            parser.print_help()

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
