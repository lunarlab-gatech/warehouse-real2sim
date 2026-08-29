#!/usr/bin/env python3
# Convert an Open3D binary PLY (mesh OR point cloud, with vertex colors) to USD for Isaac Sim.
# Deps: numpy + pxr(USD) only -- both ship inside Isaac Sim's python. No pip needed there;
# on a Mac: python3 -m venv venv && venv/bin/pip install usd-core numpy
#
#   python ply2usd.py in.ply out.usd
#   python ply2usd.py in.ply --readtest      # parse only, print counts (numpy-only sanity check)
#
# Memory note: written to handle multi-GB meshes on a small-RAM machine -- the raw
# file buffer is freed before the USD phase, which needs its own copies, so peak
# usage stays near the input file size plus the USD library's working copies.
import os
import sys

import numpy as np

PLY_NP = {'char':'i1','uchar':'u1','uint8':'u1','int8':'i1','short':'i2','ushort':'u2',
          'int':'i4','uint':'u4','int32':'i4','uint32':'u4','float':'f4','double':'f8',
          'float32':'f4','float64':'f8','int16':'i2','uint16':'u2'}


def read_ply(path):
    """Parse an Open3D binary PLY into compact arrays and FREE the raw buffer.

    Returns (pts float32 Nx3, cols uint8/float Nx3 or None, faces int32 Mx3 or None)."""
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'ply', "not a PLY file"
        fmt = f.readline().split()[1]
        endian = '<' if b'little' in fmt else '>'
        assert b'binary' in fmt, "this reader only handles binary PLY"
        elements, cur = [], None
        while True:
            ln = f.readline()
            if ln.strip() == b'end_header':
                break
            t = ln.split()
            if t[0] == b'element':
                cur = (t[1].decode(), int(t[2]), []); elements.append(cur)
            elif t[0] == b'property':
                if t[1] == b'list':
                    cur[2].append(('list', t[2].decode(), t[3].decode(), t[4].decode()))
                else:
                    cur[2].append(('scalar', t[1].decode(), t[2].decode()))
        buf = f.read()

    pts = cols = faces = None
    pos = 0
    for name, count, props in elements:
        if all(p[0] == 'scalar' for p in props):                       # vertex etc.
            dt = np.dtype([(p[2], endian + PLY_NP[p[1]]) for p in props])
            arr = np.frombuffer(buf, dtype=dt, count=count, offset=pos)
            pos += arr.nbytes
            if name == 'vertex':
                # materialize small compact copies so the giant buf can be freed
                pts = np.empty((count, 3), np.float32)
                for k, ax in enumerate(('x', 'y', 'z')):
                    pts[:, k] = arr[ax]
                if 'red' in dt.names:
                    cdt = arr['red'].dtype
                    cols = np.empty((count, 3), cdt)
                    for k, ch in enumerate(('red', 'green', 'blue')):
                        cols[:, k] = arr[ch]
            del arr
        else:                                                          # face list (assume tris)
            lp = [p for p in props if p[0] == 'list'][0]
            cnt_dt, idx_dt = endian + PLY_NP[lp[1]], endian + PLY_NP[lp[2]]
            n0 = int(np.frombuffer(buf, dtype=cnt_dt, count=1, offset=pos)[0])
            rec = np.dtype([('n', cnt_dt)] + [('i%d' % k, idx_dt) for k in range(n0)])
            arr = np.frombuffer(buf, dtype=rec, count=count, offset=pos)
            assert (arr['n'] == n0).all(), "non-triangular faces; needs a general parser"
            pos += arr.nbytes
            faces = np.empty((count, n0), np.int32)
            for k in range(n0):
                faces[:, k] = arr['i%d' % k]
            del arr
    del buf                                                            # free the raw file bytes
    return pts, cols, faces


def color_to_f32(cols):
    """uchar colors -> /255; float colors -> clipped to [0,1]."""
    if cols.dtype == np.uint8:
        return (cols.astype(np.float32) / 255.0)
    return np.clip(cols.astype(np.float32), 0.0, 1.0)


pts, cols, faces = read_ply(sys.argv[1])
print("READ OK: verts %d, faces %d, color %s"
      % (len(pts), 0 if faces is None else len(faces), cols is not None), flush=True)
if len(sys.argv) > 2 and sys.argv[2] == '--readtest':
    sys.exit(0)

from pxr import Usd, UsdGeom, Vt, Sdf

usd = sys.argv[2]
if os.path.exists(usd):
    os.remove(usd)                                   # CreateNew refuses an existing layer
stage = Usd.Stage.CreateNew(usd)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)      # PINGS/LiDAR data is Z-up
UsdGeom.SetStageMetersPerUnit(stage, 1.0)            # PLY is in meters
world = UsdGeom.Xform.Define(stage, '/World')
stage.SetDefaultPrim(world.GetPrim())                # makes the file reference-able

extent = Vt.Vec3fArray.FromNumpy(np.stack([pts.min(axis=0), pts.max(axis=0)]).astype(np.float32))
if faces is not None and len(faces):
    m = UsdGeom.Mesh.Define(stage, '/World/mesh')
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), faces.shape[1], np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.ravel())))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)   # polygon mesh, no subdivision
    m.CreateDoubleSidedAttr(True)   # interior scan: visible from both sides, no backface culling
    prim = m
else:
    p = UsdGeom.Points.Define(stage, '/World/points')
    p.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    p.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.full(len(pts), 0.02, np.float32)))  # 2 cm dots
    prim = p
prim.CreateExtentAttr(extent)                        # authored bbox: no recompute at load
if cols is not None:
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        'displayColor', Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec3fArray.FromNumpy(color_to_f32(cols)))
stage.GetRootLayer().Save()
print("wrote %s" % usd, flush=True)
