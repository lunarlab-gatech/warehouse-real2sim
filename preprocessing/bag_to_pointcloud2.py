#!/usr/bin/env python3
"""Convert a Livox CustomMsg lidar topic in a ROS 1 bag to sensor_msgs/PointCloud2.

Foxglove's 3D panel renders sensor_msgs/PointCloud2 (and LaserScan /
foxglove.PointCloud), but NOT livox_ros_driver2/msg/CustomMsg — which is why the
Livox lidar never shows up while the camera topics do. This script reads a ROS 1
.bag, re-encodes each Livox scan as a PointCloud2 (x, y, z, intensity), and
writes a NEW .bag you can open directly in Foxglove.

Pure Python (rosbags + numpy) — no ROS, no livox_ros_driver2, no colcon. The
Livox message definition is read straight out of the bag, so the custom type
deserializes without any package installed.

Usage:
    python bag_to_pointcloud2.py INBAG OUTBAG \\
        [--lidar-topic /livox/lidar] [--out-topic /livox/points] \\
        [--passthrough none|cameras|all] [--frame-id livox_frame]

    # lidar only (small, ~0.3 KB/point/scan; e.g. ~220 MB for 688 scans):
    python bag_to_pointcloud2.py in.bag out_pcl2.bag

    # lidar + camera images in one file (much larger):
    python bag_to_pointcloud2.py in.bag out_full.bag --passthrough cameras

Then in Foxglove: open OUTBAG, select the 3D panel, add a Point Cloud layer on
--out-topic (default /livox/points). If nothing shows, set the panel's display /
fixed frame to the cloud's frame_id (default "livox_frame").
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

FLOAT32 = 7        # sensor_msgs/PointField.FLOAT32
POINT_STEP = 16    # four float32 fields: x, y, z, intensity


def build_fields(typestore):
    point_field = typestore.types["sensor_msgs/msg/PointField"]
    return [
        point_field(name=name, offset=off, datatype=FLOAT32, count=1)
        for name, off in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))
    ]


def custommsg_to_xyzi(msg) -> np.ndarray:
    """[N, 4] float32 (x, y, z, intensity=reflectivity) from a Livox CustomMsg."""
    pts = msg.points
    if not len(pts):
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(
        [(p.x, p.y, p.z, float(p.reflectivity)) for p in pts], dtype=np.float32
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Livox CustomMsg -> sensor_msgs/PointCloud2 bag converter"
    )
    ap.add_argument("inbag", type=Path)
    ap.add_argument("outbag", type=Path)
    ap.add_argument("--lidar-topic", default="/livox/lidar")
    ap.add_argument("--out-topic", default="/livox/points")
    ap.add_argument(
        "--passthrough",
        choices=["none", "cameras", "all"],
        default="none",
        help="also copy other topics into the output bag: 'none' (default) = "
        "lidar only; 'cameras' = Image/CompressedImage topics too; 'all' = "
        "every non-lidar topic (includes the bulky IMU streams).",
    )
    ap.add_argument(
        "--frame-id",
        default=None,
        help="override the PointCloud2 frame_id (default: keep the source's).",
    )
    args = ap.parse_args()

    if not args.inbag.exists():
        print(f"input bag not found: {args.inbag}", file=sys.stderr)
        return 1
    if args.outbag.exists():
        print(f"output already exists, refusing to overwrite: {args.outbag}", file=sys.stderr)
        return 1
    args.outbag.parent.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    point_cloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]
    header_t = typestore.types["std_msgs/msg/Header"]
    time_t = typestore.types["builtin_interfaces/msg/Time"]
    fields = build_fields(typestore)

    with AnyReader([args.inbag]) as reader:
        all_topics = sorted({c.topic for c in reader.connections})
        lidar_conns = [c for c in reader.connections if c.topic == args.lidar_topic]
        if not lidar_conns:
            print(
                f"lidar topic {args.lidar_topic!r} not found. topics: {all_topics}",
                file=sys.stderr,
            )
            return 1

        def want_passthrough(conn) -> bool:
            if conn.topic == args.lidar_topic:
                return False
            if args.passthrough == "none":
                return False
            if args.passthrough == "all":
                return True
            return "Image" in conn.msgtype  # "cameras"

        pass_conns = [c for c in reader.connections if want_passthrough(c)]
        total_lidar = sum(c.msgcount for c in lidar_conns)

        print(f"input : {args.inbag}")
        print(f"lidar : {args.lidar_topic} ({total_lidar} msgs) -> {args.out_topic} (PointCloud2)")
        if pass_conns:
            print("copy  : " + ", ".join(
                f"{c.topic} [{c.msgtype.split('/')[-1]}]" for c in pass_conns
            ))
        print(f"output: {args.outbag}")

        with Writer(args.outbag) as writer:
            out_lidar = writer.add_connection(
                args.out_topic, point_cloud2.__msgtype__, typestore=typestore
            )
            out_pass = {
                c.id: writer.add_connection(c.topic, c.msgtype, typestore=typestore)
                for c in pass_conns
            }

            done = 0
            for conn, timestamp, raw in reader.messages(connections=lidar_conns + pass_conns):
                if conn.topic == args.lidar_topic:
                    msg = reader.deserialize(raw, conn.msgtype)
                    xyzi = custommsg_to_xyzi(msg)
                    n = int(xyzi.shape[0])
                    stamp = time_t(
                        sec=int(msg.header.stamp.sec),
                        nanosec=int(msg.header.stamp.nanosec),
                    )
                    hdr = header_t(
                        seq=done,
                        stamp=stamp,
                        frame_id=args.frame_id or msg.header.frame_id or "livox_frame",
                    )
                    cloud = point_cloud2(
                        header=hdr,
                        height=1,
                        width=n,
                        fields=fields,
                        is_bigendian=False,
                        point_step=POINT_STEP,
                        row_step=POINT_STEP * n,
                        data=np.ascontiguousarray(xyzi).reshape(-1).view(np.uint8),
                        is_dense=True,
                    )
                    writer.write(
                        out_lidar,
                        timestamp,
                        typestore.serialize_ros1(cloud, point_cloud2.__msgtype__),
                    )
                    done += 1
                    if done % 50 == 0 or done == total_lidar:
                        print(f"  lidar {done}/{total_lidar}", flush=True)
                else:
                    writer.write(out_pass[conn.id], timestamp, raw)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
