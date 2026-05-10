"""
Patch vendor VGGT files for Apple Silicon MPS compatibility and fp16 support.

Run `vggt patch` after cloning or after `git pull` on vendor submodule.
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
    print("  ✓ utils.py patched (sincos pos embed dtype)")
    return True


def _patch_vision_transformer() -> bool:
    """Patch DINOv2 ViT to avoid forced .float() on pos_embed in forward pass"""
    filepath = VENDOR_DIR / "vggt" / "layers" / "vision_transformer.py"
    with open(filepath) as f:
        content = f.read()

    if "self.pos_embed if self.pos_embed.dtype" in content:
        print("  ✓ vision_transformer.py already patched")
        return False

    old = "        pos_embed = self.pos_embed.float()"
    new = "        pos_embed = self.pos_embed if self.pos_embed.dtype == torch.float else self.pos_embed.float()"
    content = content.replace(old, new)

    with open(filepath, "w") as f:
        f.write(content)
    print("  ✓ vision_transformer.py patched (pos_embed dtype guard)")
    return True


def _patch_rope() -> bool:
    """Patch RoPE to avoid forced .float() on frequency cache"""
    filepath = VENDOR_DIR / "vggt" / "layers" / "rope.py"
    with open(filepath) as f:
        content = f.read()

    if "to(dtype=torch.float)" in content:
        print("  ✓ rope.py already patched")
        return False

    old = "exponents = torch.arange(0, dim, 2, device=device).float() / dim"
    new = "exponents = torch.arange(0, dim, 2, device=device).to(dtype=torch.float) / dim"
    content = content.replace(old, new)

    with open(filepath, "w") as f:
        f.write(content)
    print("  ✓ rope.py patched (frequency dtype)")
    return True


def _patch_track_utils(filepath) -> bool:
    """Patch tracking utils to respect input dtype instead of hardcoding float32/double"""
    with open(filepath) as f:
        content = f.read()

    fn_name = filepath.stem
    if "dtype=xy.dtype" in content:
        print(f"  ✓ {fn_name}.py already patched")
        return False

    old1 = "omega = torch.arange(embed_dim // 2, dtype=torch.double)"
    new1 = "omega = torch.arange(embed_dim // 2, dtype=torch.float64)"
    content = content.replace(old1, new1)

    old2 = "    return emb[None].float()"
    new2 = "    return emb[None]"
    content = content.replace(old2, new2)

    old3 = "div_term = (torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))"
    new3 = "div_term = (torch.arange(0, C, 2, device=xy.device, dtype=xy.dtype) * (1000.0 / C)).reshape(1, 1, int(C / 2))"
    content = content.replace(old3, new3)

    old4 = "pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)"
    new4 = "pe_x = torch.zeros(B, N, C, device=xy.device, dtype=xy.dtype)"
    content = content.replace(old4, new4)

    old5 = "pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)"
    new5 = "pe_y = torch.zeros(B, N, C, device=xy.device, dtype=xy.dtype)"
    content = content.replace(old5, new5)

    with open(filepath, "w") as f:
        f.write(content)
    print(f"  ✓ {fn_name}.py patched (fp16 dtype respect)")
    return True


def _patch_vggt_file() -> bool:
    """Patch VGGT to remove autocast wrapper and support progress_callback"""
    filepath = VENDOR_DIR / "vggt" / "models" / "vggt.py"
    with open(filepath) as f:
        content = f.read()

    patched_any = False

    # Remove autocast wrapper around heads (dead code without CUDA autocast)
    if "with torch.cuda.amp.autocast(enabled=False):" in content:
        old_block = """        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf"""
        new_block = """        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list
            
        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf"""
        content = content.replace(old_block, new_block)
        patched_any = True
        print("  ✓ vggt.py patched (removed autocast wrapper)")

    # Also handle already-patched device-agnostic autocast
    if "with torch.amp.autocast(device_type=images.device.type, enabled=False):" in content:
        old_block2 = """        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf"""
        content = content.replace(old_block2, new_block)
        patched_any = True
        print("  ✓ vggt.py patched (removed autocast wrapper, dev-agnostic variant)")

    if "progress_callback" not in content:
        content = content.replace(
            "def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):",
            "def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, progress_callback=None):",
        )
        content = content.replace(
            "aggregated_tokens_list, patch_start_idx = self.aggregator(images)",
            "aggregated_tokens_list, patch_start_idx = self.aggregator(images, progress_callback=progress_callback)",
        )
        patched_any = True
        print("  ✓ vggt.py patched (progress_callback)")

    if patched_any:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    print("  ✓ vggt.py already patched")
    return False


def _patch_aggregator_file() -> bool:
    """Patch aggregator to support progress_callback"""
    filepath = VENDOR_DIR / "vggt" / "models" / "aggregator.py"
    with open(filepath) as f:
        content = f.read()

    if "progress_callback" in content:
        print("  ✓ aggregator.py already patched")
        return False

    content = content.replace(
        "def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:",
        "def forward(self, images: torch.Tensor, progress_callback=None) -> Tuple[List[torch.Tensor], int]:",
    )
    content = content.replace(
        "for _ in range(self.aa_block_num):",
        "for step_idx in range(self.aa_block_num):",
    )
    content = content.replace(
        "                output_list.append(concat_inter)\n\n        del concat_inter",
        "                output_list.append(concat_inter)\n\n            if progress_callback is not None:\n                progress_callback(step_idx + 1, self.aa_block_num)\n\n        del concat_inter",
    )

    with open(filepath, "w") as f:
        f.write(content)
    print("  ✓ aggregator.py patched (progress_callback)")
    return True


def apply_vendor_patches():
    """Apply all required vendor patches for MPS compatibility and fp16 support"""
    print("🔧 Patching vendor VGGT for Apple Silicon MPS...")
    patched_any = False
    patched_any |= _patch_vision_transformer()
    patched_any |= _patch_rope()
    patched_any |= _patch_utils_file()
    patched_any |= _patch_vggt_file()
    patched_any |= _patch_aggregator_file()
    patched_any |= _patch_track_utils(VENDOR_DIR / "vggt" / "heads" / "track_modules" / "utils.py")
    patched_any |= _patch_track_utils(VENDOR_DIR / "vggt" / "dependency" / "track_modules" / "utils.py")
    if not patched_any:
        print("   All patches already applied.")
    else:
        print("✅ Vendor patches applied successfully.")
