# Driver image for this repository.
#
# Build from the repository root (the folder that contains start_console.sh):
#   ./controller/docker/build.sh
#
# The image is Ubuntu 22.04 + ROS 2 Humble. It does not include the Unitree
# phone app, a GPU, or a language model. The web console runs on top of it.
#
# What is installed, and why:
#   ROS 2 Humble          the driver publishes lidar, camera, IMU, and joints
#   go2_interfaces        message types, copied from controller/ in this repo
#   go2_robot_sdk         WebRTC driver, copied from controller/ in this repo
#   lidar_processor       started by robot.launch.py; it is not in this repo,
#                         so the build takes that one package from
#                         https://github.com/abizovnuralem/go2_ros2_sdk
#   aiortc, pycryptodome  the encrypted WebRTC session to the dog
#   wasmtime              decodes the dog's lidar voxels
#   python3-opencv        the dashboard's camera and map images

FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    ROBOT_IP="" \
    CONN_TYPE=webrtc \
    ROS_DOMAIN_ID=78 \
    MAP_NAME=lab

# apt packages. The launch file looks up foxglove, slam, and nav2 by name
# even when start_console.sh turns those nodes off, so they have to be present.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        python3-dev \
        python3-numpy \
        python3-yaml \
        python3-opencv \
        gcc \
        pkg-config \
        ffmpeg \
        libavdevice-dev \
        ros-humble-cv-bridge \
        ros-humble-robot-state-publisher \
        ros-humble-pointcloud-to-laserscan \
        ros-humble-joy \
        ros-humble-teleop-twist-joy \
        ros-humble-twist-mux \
        ros-humble-foxglove-bridge \
        ros-humble-slam-toolbox \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-rmw-cyclonedds-cpp \
        iputils-ping \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Do not pip-install numpy. ROS Humble is built against Ubuntu's numpy,
# and a newer wheel breaks cv_bridge.
COPY controller/docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /ros2_ws/src

# Only lidar_processor comes from upstream. Our driver and messages replace
# the copies in that repository.
RUN git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/abizovnuralem/go2_ros2_sdk.git /tmp/upstream \
    && git -C /tmp/upstream sparse-checkout set lidar_processor \
    && mv /tmp/upstream/lidar_processor /ros2_ws/src/lidar_processor \
    && rm -rf /tmp/upstream

COPY controller/go2_interfaces /ros2_ws/src/go2_interfaces
COPY controller/go2_robot_sdk /ros2_ws/src/go2_robot_sdk

WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh \
    && rosdep init || true \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && colcon build --symlink-install

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
    && echo 'source /ros2_ws/install/setup.bash' >> /root/.bashrc \
    && echo 'export ROS_DOMAIN_ID=78' >> /root/.bashrc \
    && echo 'export CONN_TYPE=webrtc' >> /root/.bashrc

COPY controller/docker/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "go2_robot_sdk", "robot.launch.py", \
     "rviz2:=false", "nav2:=false", "slam:=false", \
     "foxglove:=false", "joystick:=false", "teleop:=false"]
