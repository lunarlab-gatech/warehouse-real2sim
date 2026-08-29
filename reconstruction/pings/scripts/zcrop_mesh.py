#!/usr/bin/env python3
# World-frame Z crop of a mesh: keep triangles fully below z<=zmax, drop the rest.
# The pipeline's process.max_z_m crops in the TILTED LiDAR sensor frame (LiDAR z-axis
# is ~0.88-0.96 world-up), so it can't act as a world-height crop -- and it can't touch
# SDF-extrapolated geometry above the raw points at all. This post-hoc world-Z crop does
# both, and since ~0% of LiDAR sits above ~30 m it removes spikes without hurting completeness.
#   python3 scripts/zcrop_mesh.py <src.ply> <zmax_m> <dst.ply>
import sys
import open3d as o3d


def main():
    src, zmax, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    m = o3d.io.read_triangle_mesh(src)
    aabb = m.get_axis_aligned_bounding_box()
    mn, mx = aabb.get_min_bound(), aabb.get_max_bound()
    box = o3d.geometry.AxisAlignedBoundingBox(mn, [mx[0], mx[1], zmax])
    mc = m.crop(box)
    o3d.io.write_triangle_mesh(dst, mc)
    print(f"[zcrop] {src} z<={zmax} -> {dst}  verts {len(mc.vertices)} faces {len(mc.triangles)}", flush=True)


if __name__ == "__main__":
    main()
