#!/usr/bin/env python3
# Merge a Manifold scene's sequential ROS1 bag parts (BAG_*.bag) into ONE ROS2
# (sqlite3) bag for the lio_sam_mid360 pipeline, doing the livox_converter's
# CustomMsg -> PointCloud2 conversion OFFLINE (the runtime python converter is a
# per-point loop and a real-time bottleneck; params.yaml already expects the
# converted topic). Output topics:
#   /livox/pointcloud  sensor_msgs/msg/PointCloud2, fields x,y,z,intensity,time
#                      (all float32; time = per-point offset_time/1e9 seconds,
#                      the exact layout livox_converter.py emits)
#   /livox/imu         sensor_msgs/msg/Imu passthrough
#
#   python3 merge_manifold_ros1_to_ros2.py <scene_dir> <out_bag_dir> [--max-lidar N]
import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

LIDAR_TOPIC = "/livox/lidar"
IMU_TOPIC = "/livox/imu"
OUT_CLOUD_TOPIC = "/livox/pointcloud"
FRAME_ID = "lidar_link"  # matches params.yaml lidarFrame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", help="dir with BAG_*.bag ROS1 parts")
    ap.add_argument("out", help="output ROS2 bag directory (must not exist)")
    ap.add_argument("--max-lidar", type=int, default=0, help="0 = all (smoke tests)")
    a = ap.parse_args()

    bags = sorted(Path(a.scene_dir).glob("BAG_*.bag"))
    if not bags:
        sys.exit(f"ERROR: no BAG_*.bag in {a.scene_dir}")
    print(f"merging {len(bags)} ROS1 parts -> {a.out}")

    ts2 = get_typestore(Stores.ROS2_HUMBLE)
    PointCloud2 = ts2.types["sensor_msgs/msg/PointCloud2"]
    PointField = ts2.types["sensor_msgs/msg/PointField"]
    Header = ts2.types["std_msgs/msg/Header"]
    Time = ts2.types["builtin_interfaces/msg/Time"]
    Imu = ts2.types["sensor_msgs/msg/Imu"]
    Quaternion = ts2.types["geometry_msgs/msg/Quaternion"]
    Vector3 = ts2.types["geometry_msgs/msg/Vector3"]

    FLOAT32 = 7  # PointField datatype constant
    fields = [
        PointField(name="x", offset=0, datatype=FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=FLOAT32, count=1),
        PointField(name="time", offset=16, datatype=FLOAT32, count=1),
    ]
    POINT_STEP = 20

    # Humble writes rosbag2 metadata version <=5; newer rosbags default higher
    # (type_description_hash etc.) which Humble's ros2 bag can refuse to read.
    try:
        writer = Writer(a.out, version=5)
    except TypeError:
        print("WARNING: this rosbags version has no Writer(version=) — using default "
              "(if Humble rejects the bag, run: ros2 bag reindex <dir> sqlite3)")
        writer = Writer(a.out)

    n_lidar = n_imu = 0
    last_t = -1
    with AnyReader(bags) as r, writer:
        wc_cloud = writer.add_connection(OUT_CLOUD_TOPIC, PointCloud2.__msgtype__, typestore=ts2)
        wc_imu = writer.add_connection(IMU_TOPIC, Imu.__msgtype__, typestore=ts2)
        cons = [c for c in r.connections if c.topic in (LIDAR_TOPIC, IMU_TOPIC)]
        for con, t_ns, raw in r.messages(connections=cons):
            if t_ns < last_t:
                sys.exit(f"ERROR: non-monotonic bag record time at {t_ns}")
            last_t = t_ns
            m = r.deserialize(raw, con.msgtype)
            hdr = Header(stamp=Time(sec=int(m.header.stamp.sec),
                                    nanosec=int(m.header.stamp.nanosec)),
                         frame_id=FRAME_ID)
            if con.topic == LIDAR_TOPIC:
                if a.max_lidar and n_lidar >= a.max_lidar:
                    continue
                pts = m.points
                n = len(pts)
                arr = np.fromiter(
                    (v for p in pts for v in
                     (p.x, p.y, p.z, float(p.reflectivity), float(p.offset_time) * 1e-9)),
                    dtype=np.float32, count=n * 5).reshape(-1, 5)
                cloud = PointCloud2(
                    header=hdr, height=1, width=n, fields=fields,
                    is_bigendian=False, point_step=POINT_STEP, row_step=POINT_STEP * n,
                    data=arr.reshape(-1).view(np.uint8), is_dense=True)
                writer.write(wc_cloud, t_ns, ts2.serialize_cdr(cloud, PointCloud2.__msgtype__))
                n_lidar += 1
                if n_lidar % 500 == 0:
                    print(f"  ... {n_lidar} clouds", flush=True)
            else:
                if a.max_lidar and n_lidar >= a.max_lidar:
                    continue
                o, av, lv = m.orientation, m.angular_velocity, m.linear_acceleration
                imu = Imu(
                    header=Header(stamp=hdr.stamp, frame_id="imu_link"),
                    orientation=Quaternion(x=o.x, y=o.y, z=o.z, w=o.w),
                    orientation_covariance=np.asarray(m.orientation_covariance, np.float64),
                    angular_velocity=Vector3(x=av.x, y=av.y, z=av.z),
                    angular_velocity_covariance=np.asarray(m.angular_velocity_covariance, np.float64),
                    linear_acceleration=Vector3(x=lv.x, y=lv.y, z=lv.z),
                    linear_acceleration_covariance=np.asarray(m.linear_acceleration_covariance, np.float64))
                writer.write(wc_imu, t_ns, ts2.serialize_cdr(imu, Imu.__msgtype__))
                n_imu += 1

    print(f"DONE: {n_lidar} clouds + {n_imu} imu msgs -> {a.out}")


if __name__ == "__main__":
    main()
