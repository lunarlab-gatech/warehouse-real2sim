# Fast-LIVO2 base image (ROS Noetic + Sophus + rpg_vikit) for the GeoScan LIO
# pipeline. Reproduces the lab's `lunarlab/fastlivo2:base` (which was a local-only
# tag on the build host, never pushed to a registry) so NO image transfer is needed.
# Self-contained: public ROS base + GitHub clones; the build context is empty
# (see .dockerignore) -- FAST-LIVO2 itself is built at runtime via catkin_make
# against the bind-mounted repo, not baked into this image.
#   docker build -t lunarlab/fastlivo2:base .
# NOTE: Sophus build parallelism capped at -j8 for shared-server etiquette
#       (the original Dockerfile used -j$(nproc) = all cores).
FROM osrf/ros:noetic-desktop-full

RUN rm -f /etc/apt/sources.list.d/ros*.list

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
      cmake \
      libgoogle-glog-dev \
      libgflags-dev \
      tmux \
      neovim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && \
    git checkout a621ff && \
    find sophus -name "*.cpp" -exec sed -i 's|\.real() = \([^;]*\);|.real(\1);|g' {} \; && \
    find sophus -name "*.cpp" -exec sed -i 's|\.imag() = \([^;]*\);|.imag(\1);|g' {} \; && \
    mkdir build && cd build && \
    cmake .. && \
    make -j8 && \
    make install && \
    ldconfig && \
    mkdir -p /usr/local/share/sophus/cmake && \
    printf 'set(Sophus_INCLUDE_DIRS "/usr/local/include")\nset(Sophus_INCLUDE_DIR  "/usr/local/include")\nset(Sophus_LIBRARIES    "/usr/local/lib/libSophus.so")\nset(Sophus_LIBS         "/usr/local/lib/libSophus.so")\nset(Sophus_FOUND        TRUE)\n' > /usr/local/share/sophus/cmake/SophusConfig.cmake && \
    cd / && rm -rf /tmp/Sophus

WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/xuankuzcr/rpg_vikit.git

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash 2>/dev/null || true" >> /root/.bashrc

WORKDIR /root/catkin_ws
CMD ["/bin/bash"]
