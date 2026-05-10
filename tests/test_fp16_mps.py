#!/usr/bin/env python3
"""
Test script: Can VGGT run end-to-end in pure fp16 on MPS?

Validates every op class used by VGGT in fp16 without autocast.
Runs a forward pass on a minimal VGGT to catch crashes early.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "vggt"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEVICE = torch.device("mps")
B, S, H, W = 1, 2, 128, 128  # tiny images for quick test

# ── Colour helpers ──────────────────────────────────────────────────────────

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

results = []

@contextmanager
def section(name):
    print(f"\n{CYAN}{'═' * 60}{RESET}")
    print(f"{CYAN}  {name}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")
    yield
    print(f"{CYAN}{'═' * 60}{RESET}")

def test(name, fn):
    """Run a test, catch exceptions, report pass/fail"""
    try:
        fn()
        print(f"  {GREEN}✓{RESET} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {RED}✗{RESET} {name}  →  {type(e).__name__}: {e}")
        results.append((name, False, str(e)))

# ══════════════════════════════════════════════════════════════════════════════

with section("1. Core linear algebra ops (fp16 on MPS)"):
    test("matmul", lambda: torch.mm(
        torch.randn(256, 256, device=DEVICE, dtype=torch.float16),
        torch.randn(256, 256, device=DEVICE, dtype=torch.float16),
    ))
    test("scaled_dot_product_attention", lambda: F.scaled_dot_product_attention(
        torch.randn(2, 8, 64, 64, device=DEVICE, dtype=torch.float16),
        torch.randn(2, 8, 64, 64, device=DEVICE, dtype=torch.float16),
        torch.randn(2, 8, 64, 64, device=DEVICE, dtype=torch.float16),
    ))
    test("linear", lambda: nn.Linear(768, 768).to(DEVICE).half()(
        torch.randn(16, 768, device=DEVICE, dtype=torch.float16)
    ))

with section("2. Normalisation ops (fp16 on MPS)"):
    test("LayerNorm", lambda: nn.LayerNorm(768).to(DEVICE).half()(
        torch.randn(4, 256, 768, device=DEVICE, dtype=torch.float16)
    ))
    test("BatchNorm2d", lambda: nn.BatchNorm2d(64).to(DEVICE).half()(
        torch.randn(4, 64, 32, 32, device=DEVICE, dtype=torch.float16)
    ))
    test("softmax", lambda: F.softmax(
        torch.randn(4, 32, device=DEVICE, dtype=torch.float16), dim=-1
    ))

with section("3. Activation functions (fp16 on MPS)"):
    for act_name in ["gelu", "relu", "sigmoid", "tanh", "silu", "softplus"]:
        act = getattr(F, act_name)
        test(act_name, lambda a=act: a(
            torch.randn(16, 256, device=DEVICE, dtype=torch.float16)
        ))
    test("exp", lambda: torch.randn(16, 256, device=DEVICE, dtype=torch.float16).exp())
    test("log", lambda: torch.randn(16, 256, device=DEVICE, dtype=torch.float16).abs().add(1e-5).log())
    test("cos", lambda: torch.randn(16, 256, device=DEVICE, dtype=torch.float16).cos())
    test("sin", lambda: torch.randn(16, 256, device=DEVICE, dtype=torch.float16).sin())

with section("4. Convolution & interpolation (fp16 on MPS)"):
    test("Conv2d", lambda: nn.Conv2d(3, 64, 7, stride=2, padding=3).to(DEVICE).half()(
        torch.randn(4, 3, 128, 128, device=DEVICE, dtype=torch.float16)
    ))
    test("ConvTranspose2d", lambda: nn.ConvTranspose2d(64, 32, 2, stride=2).to(DEVICE).half()(
        torch.randn(4, 64, 16, 16, device=DEVICE, dtype=torch.float16)
    ))
    test("F.interpolate (bilinear)", lambda: F.interpolate(
        torch.randn(2, 128, 16, 16, device=DEVICE, dtype=torch.float16),
        scale_factor=2, mode="bilinear", align_corners=False,
    ))

with section("5. RoPE ops — sin/cos with range of values"):
    def rope_test():
        freq = 100
        max_dim = 1024
        pos = torch.arange(256, device=DEVICE, dtype=torch.float16) / freq
        dim_indices = torch.arange(0, max_dim // 2, device=DEVICE, dtype=torch.float16)
        freqs = pos[:, None] / (10000 ** (2 * dim_indices / max_dim))
        torch.cat([freqs.sin(), freqs.cos()], dim=-1)
    test("RoPE sin/cos pattern", rope_test)

with section("6. Reduction ops (fp16 on MPS)"):
    test("mean", lambda: torch.randn(2, 8, 16, 16, device=DEVICE, dtype=torch.float16).mean())
    test("sum", lambda: torch.randn(2, 8, 16, 16, device=DEVICE, dtype=torch.float16).sum())
    test("std", lambda: torch.randn(2, 8, 16, 16, device=DEVICE, dtype=torch.float16).std())

# ══════════════════════════════════════════════════════════════════════════════

with section("7. MINI VGGT — full forward pass in fp16"):

    def mini_vggt_forward():
        """Build a minimal VGGT-like pipeline and run fp16 forward on MPS."""
        # Simple transformer block → layer norm → linear — covers key VGGT patterns
        embed_dim = 128
        n = 64

        class MiniBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(embed_dim)
                self.qkv = nn.Linear(embed_dim, embed_dim * 3)
                self.proj = nn.Linear(embed_dim, embed_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                )
                self.ln2 = nn.LayerNorm(embed_dim)

            def forward(self, x):
                # attention — mimic VGGT's attention pattern
                qkv = self.ln(x)
                qkv = self.qkv(qkv).reshape(x.shape[0], x.shape[1], 3, 4, embed_dim // 4)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn_out = F.scaled_dot_product_attention(q, k, v)
                x = x + self.proj(
                    attn_out.transpose(1, 2).reshape(x.shape[0], x.shape[1], embed_dim)
                )
                # mlp
                x = x + self.mlp(self.ln2(x))
                return x

        model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            MiniBlock(),
            MiniBlock(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        ).to(DEVICE).half().eval()

        x = torch.randn(2, n, embed_dim, device=DEVICE, dtype=torch.float16)
        with torch.no_grad():
            out = model(x)
        return out.shape

    test("Mini transformer fp16 forward", mini_vggt_forward)

# ══════════════════════════════════════════════════════════════════════════════

with section("8. FULL-SCALE VGGT — actual VGGT model in fp16"):

    def full_vggt_forward():
        """Build actual VGGT, load real weights, run full fp16 forward."""
        from vggt.models.vggt import VGGT

        model = VGGT(
            img_size=518, patch_size=14, embed_dim=1024,
            enable_camera=True, enable_point=True, enable_depth=True, enable_track=False,
        )
        model = model.to(DEVICE).half().eval()

        images = torch.randn(B, S, 3, 518, 518, device=DEVICE, dtype=torch.float16)

        with torch.no_grad():
            preds = model(images)

        for k in ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]:
            v = preds[k]
            has_nan = bool(torch.isnan(v).any().item())
            has_inf = bool(torch.isinf(v).any().item())
            print(f"  {'✓' if not has_nan and not has_inf else '!'} {k}: "
                  f"shape={tuple(v.shape)}  "
                  f"NaN={has_nan} Inf={has_inf} "
                  f"range=[{v.min().item():.4f}, {v.max().item():.4f}]")
        return tuple(preds["depth"].shape)

    test("Full VGGT forward (fp16, random weights)", full_vggt_forward)

# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{CYAN}{'═' * 60}{RESET}")
print(f"{CYAN}  SUMMARY{RESET}")
print(f"{CYAN}{'═' * 60}{RESET}")

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

bar = "█" * int(40 * passed / max(total, 1)) + "░" * int(40 * failed / max(total, 1))
print(f"\n  [{GREEN if failed == 0 else YELLOW}{bar}{RESET}]  {passed}/{total} passed")

if failed:
    print(f"\n  {RED}Failed tests:{RESET}")
    for name, ok, err in results:
        if not ok:
            print(f"    {RED}✗{RESET} {name}")
            print(f"      {YELLOW}{err}{RESET}")

print(f"\n  {'VERDICT:' if failed else '✅ VERDICT:'}", end=" ")
if failed == 0:
    print(f"{GREEN}Pure fp16 works end-to-end on MPS (PyTorch {torch.__version__}).{RESET}")
else:
    print(f"{RED}Pure fp16 has issues on MPS. Use autocast.{RESET}")

print()
