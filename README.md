# warehouse-real2sim

Read introduction slides to get an idea of the project's goals and intent: [slides](https://gtvault-my.sharepoint.com/:p:/r/personal/ali497_gatech_edu/_layouts/15/Doc.aspx?sourcedoc=%7BC995A0CC-1562-4478-8F61-AF17ED0846E4%7D&file=Real2Sim%20Project%20introduction.pptx&action=edit&mobileredirect=true&wdOrigin=APPHOME-WEB.DIRECT%2CAPPHOME-WEB.FILEBROWSER.RECENT&wdPreviousSession=32ac4c15-f481-41d2-8622-f78fb23d9f38&wdPreviousSessionSrc=AppHomeWeb&ct=1787932627189)


## Pipeline

```
rosbag ──> preprocessing ──> odometry ──> reconstruction ──> sim_export ──> simulator
                 │                                                              ▲
                 └──────> object_extraction ──> asset_generation ───────────────┘
```

## Repository structure

### preprocessing/

Turns rosbags into the sequence format that later stages read. This directory contains bag-to-sequence conversion scripts for GeoScan and Manifold devices. Additionally, this directory contains calibration information for the camera and LiDAR. Finally, this directory contains general bag utilities such as Livox CustomMsg to PointCloud2 conversion and ROS 1 to ROS 2 merging.

### odometry/

Produces a per-scan trajectory from the LiDAR and IMU streams. Holds the Docker builds and launcher configs that drive the odometry: FAST-LIVO2 in LIO-only mode (`Fast-Livo2-GeoScan`) and LIO-SAM (`lio_sam_mid360`). Pose format conversion also lives here, including keyframe densification to per-scan poses.

### reconstruction/

3D reconstruction code.

### object_extraction/

Used for finding objects from the camera streams. Object segementation is performed using YOLO11x-seg to identify and crop objects, ByteTrack keeps object instances consistent across frames, and SAM2 refines the masks. The masked images are sent into the asset generation module.

### asset_generation/

Turns the masked object image from the object_extraction module into simulation-ready articulable assets. Currently uses PhysX-Omni, which produces textured meshes along with URDF and MJCF physics descriptions.

### sim_export/

Converts reconstruction output into simulator formats. Currently PLY mesh to USD.
