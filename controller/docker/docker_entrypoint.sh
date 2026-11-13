#!/bin/bash
# Source ROS, then run whatever the container was started with.
# ROBOT_IP and CONN_TYPE come from the environment (see start_console.sh).
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
