# Perception stage image: ultralytics base (YOLO11x-seg + SAM2), nothing baked in.
#
# perception_tracks.py is bind-mounted over /workspace by _perception_box.sh,
# so code edits never require a rebuild. YOLO/SAM weights download once into
# the mounted /weights cache. The base tag is deliberately left at :latest to
# match the image the validated runs used; pin it if reproducibility across
# rebuilds matters more than tracking upstream.

FROM ultralytics/ultralytics:latest

WORKDIR /workspace
CMD ["bash"]
