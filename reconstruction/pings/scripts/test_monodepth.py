#!/usr/bin/env python3
# Probe: can we load + run metric3d_vit_small (PINGS' monodepth model)?
# Used to decide which deps the image actually needs before rebuilding.
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
for mod in ("timm", "mmcv", "mmengine"):
    try:
        m = __import__(mod)
        print(f"{mod}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod}: NOT INSTALLED ({type(e).__name__})")

print("loading metric3d_vit_small via torch.hub ...", flush=True)
model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
print("MODEL LOADED:", type(model).__name__)
model = model.cuda().eval()
try:
    x = torch.zeros(1, 3, 616, 1064).cuda()  # ViT patch-14 friendly dims
    with torch.no_grad():
        out = model.inference({"input": x})
    d = out[0] if isinstance(out, (tuple, list)) else out
    print("INFERENCE OK, depth tensor shape:", tuple(d.shape))
except Exception as e:
    print("inference-call API differs (non-fatal for the install check):", repr(e)[:200])
print("MONODEPTH_PROBE_DONE")
