# vggt-mps

Port of Facebook Research's [VGGT](https://github.com/facebookresearch/vggt) (Visual Geometry Grounded Transformer) to Apple Silicon via PyTorch's MPS backend. Takes single or multi-view images and produces depth maps, camera poses, and 3D point clouds.

| | |
|---|---|
| Version | 2.0.0 |
| Python | 3.10+ |
| Platform | macOS 13+ on Apple Silicon (M1/M2/M3) |
| Model | facebook/VGGT-1B (1B params, ~5 GB on disk) |
| License | MIT |
| PyPI | Not yet published |

## What it produces

Given N input images, VGGT predicts:

| Output | Description |
|---|---|
| Depth maps | Per-pixel depth estimation |
| Camera poses | 6-DOF camera parameters for each view |
| 3D point clouds | Dense reconstruction (exportable as PLY, OBJ, GLB) |
| Confidence maps | Per-pixel reliability scores |

## Architecture

The upstream VGGT model is a 1B-parameter transformer trained on multi-view geometry tasks. This repo wraps it with:

- MPS device detection and dtype handling (float32 for Metal compatibility)
- A sparse attention module (`vggt_sparse_attention.py`) that patches the model at runtime for O(n) memory scaling instead of O(n^2)
- A unified CLI (`vggt` command with subcommands)
- A Gradio web interface
- An MCP server for Claude Desktop integration

```
vggt-mps/
  src/
    vggt_core.py                # Core VGGT processing
    vggt_sparse_attention.py    # Runtime sparse attention patch
    config.py                   # Centralized configuration
    visualization.py            # 3D visualization
    commands/                   # CLI subcommands (demo, reconstruct, test, benchmark, web)
    utils/                      # Model loader, image utils, export
  tests/                        # MPS, sparse attention, integration tests
  vendor/vggt/                   # VGGT upstream source (git submodule)
```

## Sparse attention

The sparse attention module replaces standard O(n^2) cross-view attention with a covisibility-masked variant. No retraining required -- it patches the loaded model at runtime.

| Images | Standard memory | Sparse memory | Reduction |
|---|---|---|---|
| 100 | O(10K) | O(1K) | 10x |
| 500 | O(250K) | O(5K) | 50x |
| 1000 | O(1M) | O(10K) | 100x |

Output difference vs. standard attention: reported as 0.000000 in tests. In practice this means numerically identical within float32 precision.

## Requirements

- Apple Silicon Mac (M1, M2, or M3)
- 8 GB+ RAM
- 6 GB disk for model weights
- Python 3.10+

## Install

### From source (recommended)

```bash
git clone https://github.com/jmanhype/vggt-mps.git
cd vggt-mps
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
make install   # uses uv pip install -e .
```

### Download model weights

```bash
vggt download
# or: python main.py download
```

The model downloads from [Hugging Face](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt) (~5 GB).

## Usage

### CLI

```bash
vggt demo                              # run with sample images
vggt demo --kitchen --images 4         # kitchen dataset, 4 views
vggt reconstruct data/*.jpg            # your own images
vggt reconstruct --sparse data/*.jpg   # sparse attention for large sets
vggt reconstruct --export ply data/*.jpg
vggt web                               # launch Gradio UI
vggt web --port 8080 --share           # public link
vggt test --suite all                  # run test suite
vggt benchmark --compare               # performance comparison
```

### Python

```python
from src.vggt_sparse_attention import make_vggt_sparse

# Patch any loaded VGGT model for sparse attention
sparse_model = make_vggt_sparse(model, device="mps")
output = sparse_model(images)
```

### MCP server (Claude Desktop)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vggt-agent": {
      "command": "uv",
      "args": [
        "run", "--python", "/path/to/vggt-mps/vggt-env/bin/python",
        "--with", "fastmcp", "fastmcp", "run",
        "/path/to/vggt-mps/src/vggt_mps_mcp.py"
      ]
    }
  }
}
```

Available MCP tools: `vggt_quick_start_inference`, `vggt_extract_video_frames`, `vggt_process_images`, `vggt_create_3d_scene`, `vggt_reconstruct_3d_scene`, `vggt_visualize_reconstruction`.

## Dependencies

| Package | Role |
|---|---|
| torch >= 2.0.0 | Computation backend (MPS) |
| torchvision >= 0.15.0 | Image transforms |
| einops >= 0.6.1 | Tensor reshaping |
| transformers >= 4.30.0 | Model loading |
| huggingface-hub >= 0.16.0 | Weight download |
| timm >= 0.9.0 | Vision model components |
| opencv-python >= 4.7.0 | Image I/O |
| gradio >= 3.40.0 | Optional: web interface |
| fastmcp >= 0.1.0 | Optional: MCP server |

## Limitations

- Runs on Apple Silicon only. No CUDA path in this repo (use upstream VGGT for that).
- Uses float32 exclusively; MPS does not support float16 autocast for this model.
- The `vggt download` command pulls ~5 GB over the network with no resume support.
- Not published to PyPI yet. Install from source.
- Sparse attention memory numbers in the table above are asymptotic ratios, not measured byte counts.
- The vendored `vendor/vggt/` submodule tracks upstream facebookresearch/vggt.

## References

- [VGGT paper](https://arxiv.org/pdf/2507.04009)
- [facebook/vggt](https://github.com/facebookresearch/vggt) (upstream)
- [Hugging Face model](https://huggingface.co/facebook/VGGT-1B)

## Contributing

Development branch is `develop`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
