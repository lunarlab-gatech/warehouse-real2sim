#!/usr/bin/env python3
"""Perception stage [C]: YOLO11x-seg + ByteTrack instance masks + SAM2 best-view crops.

Reads a KITTI-style sequence dir (output of _prep_geoscan.sh) and writes the
stage-[C] artifacts frozen in pipeline/CONTRACTS.md:

  $SEQ/masks/image_N/%06d.png          uint16 PNG, 0 = background, k > 0 = global
                                       instance pixel id (one running counter across
                                       ALL cameras); one PNG per frame, count == frame
                                       count (all-zero when nothing detected); larger
                                       bbox paints last so closer objects win overlaps
  $SEQ/objects/tracks_cam{N}.json      per-instance track records (see CONTRACTS.md)
  $SEQ/objects/crops_raw/inst_<pid>.png  SAM2-refined white-background best-view crop
                                       (PhysX-Omni input; mirrors main.py's recipe)
  $SEQ/objects/perception_report.json  per-camera + global stats for validation

Tracker state is reset between cameras by constructing a fresh YOLO instance per
camera, so ByteTrack local ids never leak across image_N dirs; the (cam, local id)
pair is mapped to a globally unique pixel id on first sight.

Idempotent: a camera is skipped when its mask PNG count matches its frame count AND
tracks_cam{N}.json exists; pass --force to redo. Skipped cameras keep their pixel
ids and the global counter is seeded past them.

Runs inside the ultralytics-based docker image (see _perception_box.sh).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO

# CONTRACTS.md "Class policy": always masked, never asset-eligible.
MASK_ONLY_CLASSES = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

EDGE_MARGIN_PX = 4        # bbox within this many px of the border -> penalized view
EDGE_PENALTY = 0.5
SHARPNESS_KNEE = 100.0    # laplacian variance v -> v / (v + knee), saturating in [0, 1)
MAX_PIXEL_ID = 65535      # uint16 mask PNGs

FRAME_GLOB = "[0-9][0-9][0-9][0-9][0-9][0-9].png"


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO11x-seg tracking masks + SAM2 best-view crops (pipeline stage [C])")
    p.add_argument("--seq", required=True,
                   help="path to sequences/00 (KITTI layout from _prep_geoscan.sh)")
    p.add_argument("--cams", default="2,3,4",
                   help="comma-separated camera indices (image_N dirs), default 2,3,4")
    p.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    p.add_argument("--yolo", default="yolo11x-seg.pt",
                   help="YOLO segmentation weights (asset name or path; auto-downloads)")
    p.add_argument("--sam", default="sam2.1_l.pt",
                   help="SAM2 weights (asset name or path; auto-downloads)")
    p.add_argument("--device", default="0", help="CUDA device (inside container: 0)")
    p.add_argument("--force", action="store_true",
                   help="redo cameras even when cached masks + tracks json exist")
    return p.parse_args()


def list_frames(img_dir):
    """Sorted %06d.png frames of one camera dir."""
    return sorted(img_dir.glob(FRAME_GLOB))


def camera_is_cached(seq, cam, n_frames):
    """Cached iff mask PNG count == frame count AND tracks json exists (json is
    written last, so a partial/crashed run never looks cached)."""
    tracks_json = seq / "objects" / f"tracks_cam{cam}.json"
    if not tracks_json.is_file() or n_frames == 0:
        return False
    masks_dir = seq / "masks" / f"image_{cam}"
    n_masks = len(list(masks_dir.glob(FRAME_GLOB))) if masks_dir.is_dir() else 0
    return n_masks == n_frames


def seed_pixel_counter(obj_dir, redo_cams):
    """Highest pixel id claimed by any tracks_cam*.json we are KEEPING (skipped or
    out-of-scope cameras), so new ids never collide with cached ones."""
    max_pid = 0
    for tj in sorted(obj_dir.glob("tracks_cam*.json")):
        m = re.fullmatch(r"tracks_cam(\d+)\.json", tj.name)
        if m is None or int(m.group(1)) in redo_cams:
            continue
        try:
            instances = json.loads(tj.read_text()).get("instances", {})
        except (json.JSONDecodeError, OSError):
            continue
        for pid in instances:
            max_pid = max(max_pid, int(pid))
    return max_pid


def clamp_box(box, w, h):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def detection_score(gray, box, conf):
    """Best-view score: area * conf * sharpness * edge_margin_ok.
    sharpness = laplacian variance of the gray crop, saturating-normalized to [0,1);
    edge_margin_ok = 0.5 when the bbox touches within EDGE_MARGIN_PX of the border."""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area_frac = (x2 - x1) * (y2 - y1) / float(w * h)
    crop = gray[y1:y2, x1:x2]
    if crop.size < 4:
        sharpness = 0.0
    else:
        lap_var = cv2.Laplacian(crop, cv2.CV_64F).var()
        sharpness = lap_var / (lap_var + SHARPNESS_KNEE)
    edge_ok = EDGE_PENALTY if (x1 <= EDGE_MARGIN_PX or y1 <= EDGE_MARGIN_PX or
                               x2 >= w - EDGE_MARGIN_PX or y2 >= h - EDGE_MARGIN_PX) else 1.0
    return float(area_frac * conf * sharpness * edge_ok)


def detection_mask(result, det_idx, box, w, h):
    """Boolean HxW mask for detection det_idx. retina_masks=True asks ultralytics
    for native-resolution masks; if a model-size mask comes back anyway, resize it
    to the frame's true HxW. None masks fall back to the bbox rectangle."""
    masks = result.masks
    if masks is not None and det_idx < len(masks.data):
        m = masks.data[det_idx].cpu().numpy()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return m > 0.5
    rect = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = clamp_box(box, w, h)
    rect[y1:y2, x1:x2] = True
    return rect


def sam_white_bg_crop(sam_model, img, box, device):
    """SAM2 once on the track's best view, composite onto white, crop to bbox
    (mirrors main.py). Falls back to the raw bbox crop if SAM yields no mask."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    final_img = None
    sam_results = sam_model(img, bboxes=[[x1, y1, x2, y2]], verbose=False, device=device)
    if sam_results[0].masks is not None and len(sam_results[0].masks.xy) > 0:
        polygon = sam_results[0].masks.xy[0]
        if len(polygon) >= 3:
            mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_img, [np.array(polygon, dtype=np.int32)], 255)
            white_bg = np.full_like(img, 255)
            object_extracted = cv2.bitwise_and(img, img, mask=mask_img)
            bg_mask = cv2.bitwise_not(mask_img)
            white_bg_masked = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
            final_img = cv2.add(white_bg_masked, object_extracted)
    if final_img is None:
        final_img = img
    return final_img[y1:y2, x1:x2]


def drop_stale_outputs(seq, cam):
    """Before reprocessing a camera, remove its old tracks json and the crops it
    referenced so a --force run leaves no orphaned inst_<pid>.png files."""
    tracks_json = seq / "objects" / f"tracks_cam{cam}.json"
    if not tracks_json.is_file():
        return
    try:
        instances = json.loads(tracks_json.read_text()).get("instances", {})
    except (json.JSONDecodeError, OSError):
        instances = {}
    for rec in instances.values():
        crop = rec.get("best_crop")
        if crop:
            (seq / "objects" / crop).unlink(missing_ok=True)
    tracks_json.unlink()


def process_camera(seq, cam, frames, args, alloc_pid, sam_model):
    """Run YOLO tracking over one camera, write mask PNGs, then SAM2 crops and the
    tracks json. Returns the per-camera report entry."""
    masks_dir = seq / "masks" / f"image_{cam}"
    crops_dir = seq / "objects" / "crops_raw"
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    drop_stale_outputs(seq, cam)

    model = YOLO(args.yolo)  # fresh instance => fresh ByteTrack state per camera
    tracks = {}              # pixel id -> record (contract schema + _best_* privates)
    local2pid = {}           # ByteTrack local id -> global pixel id (this camera only)
    coverage_sum = 0.0

    for fi, fpath in enumerate(frames):
        frame_name = fpath.stem
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"unreadable frame {fpath}")
        h, w = img.shape[:2]
        result = model.track(img, persist=True, conf=args.conf, verbose=False,
                             device=args.device, tracker="bytetrack.yaml",
                             retina_masks=True)[0]
        mask_png = np.zeros((h, w), dtype=np.uint16)

        if result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            tids = result.boxes.id.cpu().numpy().astype(int)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # ascending area: larger (closer) instances paint last and win overlaps
            for di in np.argsort(areas):
                box, conf = boxes[di], float(confs[di])
                cls, tid = int(classes[di]), int(tids[di])
                if tid not in local2pid:
                    name = result.names[cls]
                    local2pid[tid] = alloc_pid()
                    tracks[local2pid[tid]] = {
                        "cam": cam, "local_track_id": tid,
                        "class_id": cls, "class_name": name,
                        "mask_only": name in MASK_ONLY_CLASSES,
                        "frames": {}, "best_frame": None, "best_crop": None,
                        "best_score": 0.0,
                    }
                pid = local2pid[tid]
                rec = tracks[pid]
                mask_png[detection_mask(result, di, box, w, h)] = pid
                x1, y1, x2, y2 = clamp_box(box, w, h)
                rec["frames"][frame_name] = [x1, y1, x2, y2, round(conf, 4)]
                score = detection_score(gray, box, conf)
                if score > rec["best_score"] or rec["best_frame"] is None:
                    rec["best_score"] = score
                    rec["best_frame"] = frame_name
                    rec["_best_img"] = img  # imread gives a fresh buffer per frame
                    rec["_best_box"] = (x1, y1, x2, y2)

        cv2.imwrite(str(masks_dir / f"{frame_name}.png"), mask_png)
        coverage_sum += np.count_nonzero(mask_png) / float(mask_png.size)
        if (fi + 1) % 50 == 0 or fi + 1 == len(frames):
            print(f"[cam{cam}] frame {fi + 1}/{len(frames)} tracks={len(tracks)}", flush=True)

    del model  # release before SAM2; next camera builds its own tracker

    # SAM2 ONCE per non-mask_only track, on the stored best view
    n_crops = 0
    for pid, rec in sorted(tracks.items()):
        if rec["mask_only"] or rec["best_frame"] is None:
            continue
        crop = sam_white_bg_crop(sam_model, rec["_best_img"], rec["_best_box"], args.device)
        if crop is None or crop.size == 0:
            print(f"[cam{cam}] WARN: empty crop for inst_{pid} ({rec['class_name']}), skipping")
            continue
        rel = f"crops_raw/inst_{pid}.png"
        cv2.imwrite(str(seq / "objects" / rel), crop)
        rec["best_crop"] = rel
        n_crops += 1
    print(f"[cam{cam}] wrote {n_crops} SAM2 crops for {len(tracks)} tracks", flush=True)

    # tracks json LAST: its presence marks the camera as complete (cache key)
    instances = {}
    for pid, rec in sorted(tracks.items()):
        pub = {k: v for k, v in rec.items() if not k.startswith("_")}
        pub["best_score"] = round(float(pub["best_score"]), 4)
        instances[str(pid)] = pub
    tracks_json = seq / "objects" / f"tracks_cam{cam}.json"
    tmp = tracks_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"instances": instances}, indent=2))
    tmp.replace(tracks_json)

    return {
        "n_frames": len(frames),
        "n_tracks": len(tracks),
        "n_crops": n_crops,
        "class_histogram": dict(Counter(r["class_name"] for r in tracks.values())),
        "mean_mask_coverage": round(coverage_sum / max(1, len(frames)), 5),
    }


def cached_camera_stats(seq, cam, n_frames):
    """Rebuild the report entry for a skipped camera from its artifacts on disk."""
    instances = json.loads(
        (seq / "objects" / f"tracks_cam{cam}.json").read_text()).get("instances", {})
    coverage_sum, n_masks = 0.0, 0
    for mpath in sorted((seq / "masks" / f"image_{cam}").glob(FRAME_GLOB)):
        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        coverage_sum += np.count_nonzero(m) / float(m.size)
        n_masks += 1
    return {
        "n_frames": n_frames,
        "n_tracks": len(instances),
        "n_crops": sum(1 for r in instances.values() if r.get("best_crop")),
        "class_histogram": dict(Counter(r["class_name"] for r in instances.values())),
        "mean_mask_coverage": round(coverage_sum / max(1, n_masks), 5),
    }


def main():
    args = parse_args()
    seq = Path(args.seq)
    if not seq.is_dir():
        sys.exit(f"ERROR: --seq dir not found: {seq}")
    cams = [int(c) for c in args.cams.split(",") if c.strip()]

    obj_dir = seq / "objects"
    obj_dir.mkdir(parents=True, exist_ok=True)

    frames_by_cam = {}
    for cam in cams:
        frames = list_frames(seq / f"image_{cam}")
        if not frames:
            sys.exit(f"ERROR: no %06d.png frames in {seq / f'image_{cam}'}")
        frames_by_cam[cam] = frames

    skip = {cam: (not args.force) and camera_is_cached(seq, cam, len(frames_by_cam[cam]))
            for cam in cams}
    redo_cams = {cam for cam in cams if not skip[cam]}

    # one global running pixel-id counter across ALL cameras (contract), seeded
    # past any ids already claimed by cached/out-of-scope tracks jsons
    counter = {"pid": seed_pixel_counter(obj_dir, redo_cams)}

    def alloc_pid():
        counter["pid"] += 1
        if counter["pid"] > MAX_PIXEL_ID:
            raise RuntimeError(f"pixel id overflow (> {MAX_PIXEL_ID}, uint16 masks)")
        return counter["pid"]

    sam_model = SAM(args.sam) if redo_cams else None

    report = {"contract_version": 1, "seq": str(seq), "conf": args.conf, "cams": {}}
    for cam in cams:
        n_frames = len(frames_by_cam[cam])
        if skip[cam]:
            print(f"[cam{cam}] cached ({n_frames} masks + tracks_cam{cam}.json); "
                  f"skipping (--force to redo)", flush=True)
            report["cams"][str(cam)] = cached_camera_stats(seq, cam, n_frames)
        else:
            print(f"[cam{cam}] processing {n_frames} frames "
                  f"(yolo={args.yolo} conf={args.conf})", flush=True)
            report["cams"][str(cam)] = process_camera(
                seq, cam, frames_by_cam[cam], args, alloc_pid, sam_model)

    per_cam = list(report["cams"].values())
    hist = Counter()
    for entry in per_cam:
        hist.update(entry["class_histogram"])
    total_frames = sum(e["n_frames"] for e in per_cam)
    report["totals"] = {
        "n_frames": total_frames,
        "n_tracks": sum(e["n_tracks"] for e in per_cam),
        "n_crops": sum(e["n_crops"] for e in per_cam),
        "class_histogram": dict(hist),
        "mean_mask_coverage": round(
            sum(e["mean_mask_coverage"] * e["n_frames"] for e in per_cam)
            / max(1, total_frames), 5),
        "max_pixel_id": counter["pid"],
    }
    (obj_dir / "perception_report.json").write_text(json.dumps(report, indent=2))
    print(f"perception complete: cams={cams} tracks={report['totals']['n_tracks']} "
          f"crops={report['totals']['n_crops']} -> {obj_dir / 'perception_report.json'}",
          flush=True)


if __name__ == "__main__":
    main()
