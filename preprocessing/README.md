# Preprocessing

General rosbag utilities used before the pipeline stages. Both scripts are pure Python (`pip install rosbags numpy`) and need no ROS installation — the Livox message definition is read straight out of the bag.

Note: the bag-to-sequence converters and the camera/LiDAR calibration definitions live with the reconstruction backend that consumes them, in `reconstruction/pings/scripts/` (`convert_geoscan_bag.py`, `convert_manifold_bags.py`, `geoscan_calib.py`, `manifold_calib.py`), because the PINGS prep launchers call them from there.

## Files

- `bag_to_pointcloud2.py` — converts a Livox CustomMsg lidar topic in a ROS 1 bag to `sensor_msgs/PointCloud2` and writes a new bag. Foxglove's 3D panel cannot render CustomMsg, so this is how you visually inspect a recording. Optional passthrough of camera topics or all other topics.

```
python bag_to_pointcloud2.py in.bag out_pcl2.bag [--passthrough cameras]
```

- `merge_manifold_ros1_to_ros2.py` — merges a Manifold scene's sequential ROS 1 bag parts (`BAG_*.bag`) into one ROS 2 (sqlite3) bag, converting CustomMsg to PointCloud2 offline. This feeds the LIO-SAM pipeline in `odometry/`: the output topics and frame id match what LIO-SAM's `params.yaml` expects, and doing the conversion offline avoids the runtime per-point converter that bottlenecks real-time playback.

```
python3 merge_manifold_ros1_to_ros2.py <scene_dir> <out_bag_dir> [--max-lidar N]
```
