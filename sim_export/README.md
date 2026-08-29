# Sim export

Converts reconstruction output into simulator formats. Currently one converter: PLY to USD for Isaac Sim.

## Files

- `ply2usd.py` — converts an Open3D binary PLY (mesh or point cloud, with vertex colors) to USD. Dependencies are numpy and `pxr` (USD) only, both of which ship inside Isaac Sim's bundled Python, so no installs are needed there; elsewhere, `pip install usd-core numpy`. Written to handle multi-GB meshes on a small-RAM machine: the raw file buffer is freed before the USD phase, so peak memory stays near the input file size plus the USD library's working copies.

## Usage

```
python ply2usd.py in.ply out.usd
```

`--readtest` parses the PLY and prints counts without writing USD (numpy-only sanity check).

Typical flow: a mesh from `reconstruction/` (PINGS or NKSR output) is converted here, then the `.usd` is transferred to the machine running Isaac Sim.
