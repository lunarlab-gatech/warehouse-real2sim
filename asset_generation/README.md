# Asset generation

Turns the object crops from `object_extraction/` into simulation-ready articulable assets: textured meshes with URDF and MJCF physics descriptions. The generation itself is done by [PhysX-Omni](https://github.com/physx-omni/PhysX-Omni).

The PhysX-Omni source is not in this repository. It is under the S-Lab License 1.0 (non-commercial), so this module ships only our own code plus a small patch, and the setup step clones upstream at a pinned commit. Model weights are downloaded from HuggingFace at setup time and never enter git.

## Files

- `_physx_assets.sh` — the pipeline launcher. `--setup` clones PhysX-Omni at pinned commit `5ba54ee`, applies the patch, builds the Docker image, and downloads the weights (the only mode that goes online). The normal mode turns a directory of crop PNGs into assets: crops are split round-robin across one or two GPUs, each shard runs the three PhysX-Omni stages in its own container inside a shadow workdir of symlinks (upstream hardcodes its output directory relative to the working directory, so shards need disjoint ones), and finished assets are harvested into the output directory. Idempotent: a crop with an existing `basic.xml` in the output is skipped.
- `Dockerfile` — builds the CUDA 11.8 / torch 2.4.0 image with the compiled dependencies (xformers, flash-attn, spconv, nvdiffrast, kaolin, and others). It contains no PhysX-Omni source; the repo is bind-mounted at run time.
- `physx_omni_lunar.patch` — 17 changed lines across two upstream files, applied by `--setup`:
  - `1vlm_demo.py`: the VLM loads with `sdpa` attention instead of `flash_attention_2`, and per-image failures print a full traceback instead of being silently swallowed.
  - `decoder_each.py`: forces the xformers attention backend, frees three unused models after pipeline load (avoids an out-of-memory crash during mesh extraction on 24 GB cards), and decodes only the mesh and gaussian formats.

## Usage

One-time setup (clone, patch, image build, weight download):

```
bash _physx_assets.sh --setup
```

Then per scene:

```
bash _physx_assets.sh <crops_dir> <out_assets_dir> <gpu> [gpu2]
```

`<crops_dir>` is the `objects/crops_final` directory produced by `object_extraction/` and must live under `ROOT` (default `/scratch/ali497`, env-overridable) so the bind mount can see it. Each finished asset lands in `<out_assets_dir>/<crop_name>/` with its mesh, textures, `basic.urdf`, and `basic.xml`.
