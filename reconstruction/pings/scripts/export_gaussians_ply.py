#!/usr/bin/env python3
"""Export PINGS gaussians -> INRIA-format 3D-Gaussian-Splatting .ply (for SuperSplat etc).

PINGS has NO native splat export: it stores neural points + MLP decoders (model/pin_map.pth)
and *spawns* gaussians on the fly at render time (gaussian_renderer.spawn_gaussians). This
script loads a run, spawns gaussians over ALL neural points (chunked, valid-filtered), bakes
colors from a single viewpoint (a .ply is view-independent; PINGS' view dependence is a small
+/-0.1 residual), and writes the standard 3DGS schema:
    x,y,z, nx,ny,nz, f_dc_0..2, opacity, scale_0..2, rot_0..3   (all float32, SH degree 0)

Run inside pings:cu118, from the project root (/packages/pings):
    python3 scripts/export_gaussians_ply.py <run_dir> [--out file.ply] [--chunk 200000] [--min-scale 0]
Then drag the .ply into https://superspl.at/editor (or any 3DGS .ply viewer).

Notes / caveats:
  * PINGS uses 2D gaussian_surfels -> the 3rd scale is ~1e-7 (flat splats). Viewers render
    them fine but very thin; pass --min-scale 0.01 to floor the thin dim if a viewer culls them.
  * opacity is PINGS' tanh-"alpha" in (0,1] (the ScaffoldGS trick); we store inverse_sigmoid(alpha).
  * quaternion order is assumed (w,x,y,z) -- the rasterizer/INRIA convention PINGS feeds directly.
  * Loading mirrors inspect_pings.py exactly so spawn params match the trained run.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.decoder import Decoder  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.tools import setup_experiment, load_decoders  # noqa: E402
from gaussian_splatting.gaussian_renderer import spawn_gaussians  # noqa: E402

C0 = 0.28209479177387814  # SH band-0 constant: f_dc = (rgb - 0.5) / C0
PLY_PROPS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def load_run(experiment_path, log_on=False):
    """Load config + neural points + decoders exactly as inspect_pings.py does."""
    yamls = glob.glob(os.path.join(experiment_path, "*.yaml"))
    assert len(yamls) == 1, f"need exactly one config yaml in {experiment_path}, found {yamls}"
    config = Config()
    config.load(yamls[0])
    config.model_path = os.path.join(experiment_path, "model", "pin_map.pth")
    config.silence = not log_on
    config.gs_on = True

    full_cfg = os.path.join(experiment_path, "meta", "config_all.yaml")
    if os.path.exists(full_cfg):
        import yaml
        a = yaml.safe_load(open(full_cfg))
        # these three are consumed by spawn_gaussians -> must match the trained run
        config.displacement_range_ratio = a["displacement_range_ratio"]
        config.max_scale_ratio = a["max_scale_ratio"]
        config.unit_scale_ratio = a["unit_scale_ratio"]

    setup_experiment(config, sys.argv, debug_mode=True)  # sets device/seed, creates NO dirs
    if getattr(config, "device", None) in (None, ""):
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gd, cd = config.feature_dim, config.color_feature_dim
    dist_dim = 1 if config.dist_concat_on else 0
    view_dim = 3 if config.view_concat_on else 0
    mlp = {
        "sdf": Decoder(config, gd, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1),
        "color": (Decoder(config, cd, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel)
                  if config.color_on else None),
        "semantic": None,
        "gauss_xyz":   Decoder(config, gd, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, 0),
        "gauss_rot":   Decoder(config, gd, config.gs_mlp_hidden_dim, config.gs_mlp_level, 4, config.spawn_n_gaussian, 0),
        "gauss_scale": Decoder(config, gd, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, 0),
        "gauss_alpha": Decoder(config, gd, config.gs_mlp_hidden_dim, config.gs_mlp_level, 1, config.spawn_n_gaussian, dist_dim),
        "gauss_color": Decoder(config, cd, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, view_dim),
    }
    loaded = torch.load(config.model_path, map_location=config.device)
    neural_points = loaded["neural_points"]
    neural_points.config = config
    neural_points.temporal_local_map_on = False
    load_decoders(loaded, mlp)
    return config, neural_points, mlp


def export(config, neural_points, mlp, out_path, chunk_size=200000, view_from=None, min_scale=0.0):
    dev = config.device
    pos = neural_points.neural_points  # (N,3) world
    N = int(pos.shape[0])
    F = int(neural_points.geo_features.shape[0])
    assert F == N + 1, f"expected geo_features = N+1 ({N + 1}), got {F}"
    has_color = neural_points.point_colors is not None

    if view_from is None:
        view_from = pos.float().mean(dim=0)  # bake colors from the scene centroid
    cam_origin = view_from.to(dev).float()

    def slc(t, m):
        return None if t is None else t[m].to(dev)

    chunks = {k: [] for k in ("xyz", "scale", "rot", "alpha", "color")}
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        mask = torch.zeros(F, dtype=torch.bool, device=pos.device)
        mask[s:e] = True
        mask[-1] = True            # trailing dummy row, stripped by spawn's geo_feature[:-1]
        mask_a = mask[:-1]         # length N (points/orientation/color have no dummy)
        data = {
            "position": pos[mask_a].to(dev),
            "orientation": neural_points.point_orientations[mask_a].to(dev),
            "geo_feature": neural_points.geo_features[mask].to(dev),
            "color_feature": neural_points.color_features[mask].to(dev),
            "resolution": neural_points.resolution,
            "valid_mask": (neural_points.valid_gs_mask[mask_a] > 0).to(dev) if neural_points.valid_gs_mask is not None else None,
            "free_mask": (neural_points.free_gs_mask[mask_a] > 0).to(dev) if neural_points.free_gs_mask is not None else None,
            "stability": slc(neural_points.point_certainties, mask_a),
        }
        if has_color:
            data["color"] = neural_points.point_colors[mask_a].to(dev)

        with torch.no_grad():
            r = spawn_gaussians(
                data, mlp, None, cam_origin,
                dist_concat_on=config.dist_concat_on,
                view_concat_on=config.view_concat_on,
                alpha_filter_on=True, scale_filter_on=False,
                learn_color_residual=config.learn_color_residual,
                gs_type=config.gs_type,
                displacement_range_ratio=config.displacement_range_ratio,
                max_scale_ratio=config.max_scale_ratio,
                unit_scale_ratio=config.unit_scale_ratio,
            )
        if r is None or r["gaussian_xyz"].shape[0] == 0:
            continue
        chunks["xyz"].append(r["gaussian_xyz"].detach().float().cpu())
        chunks["scale"].append(r["gaussian_scale"].detach().float().cpu())
        chunks["rot"].append(r["gaussian_rot"].detach().float().cpu())
        chunks["alpha"].append(r["gaussian_alpha"].detach().reshape(-1).float().cpu())
        chunks["color"].append(r["gaussian_color"].detach().float().cpu())
        print(f"[export] points {s}:{e}/{N}  +{r['gaussian_xyz'].shape[0]} gaussians", flush=True)
        del r
        if dev != "cpu":
            torch.cuda.empty_cache()

    xyz = torch.cat(chunks["xyz"]).numpy()
    scale = torch.cat(chunks["scale"]).numpy()
    rot = torch.cat(chunks["rot"]).numpy()
    alpha = torch.cat(chunks["alpha"]).numpy()
    color = torch.cat(chunks["color"]).numpy()
    write_inria_ply(out_path, xyz, color, alpha, scale, rot, min_scale)


def write_inria_ply(path, xyz, color01, alpha, scale_m, rot, min_scale=0.0):
    eps = 1e-6
    a = np.clip(alpha.astype(np.float64), eps, 1 - eps)
    opacity = np.log(a / (1 - a)).astype(np.float32).reshape(-1, 1)     # inverse sigmoid
    f_dc = ((color01.astype(np.float32) - 0.5) / C0).astype(np.float32)
    if min_scale > 0:
        scale_m = np.maximum(scale_m, min_scale)
    scale_log = np.log(np.clip(scale_m.astype(np.float64), 1e-12, None)).astype(np.float32)
    rot = rot.astype(np.float32)
    normals = np.zeros((xyz.shape[0], 3), np.float32)
    arr = np.concatenate([xyz.astype(np.float32), normals, f_dc, opacity, scale_log, rot], axis=1)
    assert arr.shape[1] == len(PLY_PROPS), (arr.shape[1], len(PLY_PROPS))

    finite = np.isfinite(arr).all(axis=1)
    if not finite.all():
        print(f"[export] dropping {int((~finite).sum())} non-finite gaussians", flush=True)
        arr = arr[finite]
    n = arr.shape[0]

    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in PLY_PROPS)
              + "end_header\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
    print(f"[export] wrote {n} gaussians -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output .ply (default: <run_dir>/gaussians_3dgs.ply)")
    ap.add_argument("--chunk", type=int, default=200000, help="neural points per spawn batch")
    ap.add_argument("--min-scale", type=float, default=0.0, help="floor the thin surfel scale (m), e.g. 0.01")
    ap.add_argument("--log-on", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "gaussians_3dgs.ply")
    config, neural_points, mlp = load_run(a.run_dir, log_on=a.log_on)
    export(config, neural_points, mlp, out, chunk_size=a.chunk, min_scale=a.min_scale)


if __name__ == "__main__":
    main()
