#!/usr/bin/env bash
# Web console. The computer and the dog must already be on the same lab Wi-Fi.
# The dog joins that Wi-Fi by itself when it powers on. This script checks that
# the dog answers, then starts the driver container and the page on port 8080.
#
#   ./start_console.sh              start, or restart the driver only if it has no data
#   ./start_console.sh --reset      always restart the driver first
#   ROBOT_IP=172.17.39.82 ./start_console.sh
#   GO2_IMAGE=go2:dev ./start_console.sh
#
# Build the image once with ./controller/docker/build.sh before the first run.
set -u
STACK=$(dirname "$(readlink -f "$0")")
DOG=${ROBOT_IP:-172.17.39.82}
C=${GO2_CONTAINER:-go2_khang}
# go2:dev is the image from ./controller/docker/build.sh. If that tag is
# missing and the older lab image khang/go2:dev is present, use that.
IMAGE=${GO2_IMAGE:-go2:dev}
NETS=${GO2_CONTROL_NETS:-127.0.0.0/8,172.17.0.0/18}
ROS='source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash'

MAP=$STACK/go2_maps/lab.grid.npz
PLACES=$STACK/go2_maps/lab.places.json
mkdir -p "$STACK/go2_maps" "$STACK/go2_recordings"
if [ -f "$MAP" ]; then
    echo "Map: $(readlink -f "$MAP")"
    echo "     $(date -r "$MAP" '+%Y-%m-%d %H:%M:%S')"
else
    echo "No saved map at $MAP. The page will start empty; use Start mapping once the dog is connected."
fi
[ -f "$PLACES" ] || echo "No places file yet ($PLACES). The page can save spots after you drive."

if ! command -v docker >/dev/null; then
    echo "Docker is not installed. On Ubuntu: sudo apt install docker.io && sudo usermod -aG docker \$USER"
    echo "Then log out and back in, and build the image: ./controller/docker/build.sh"
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [ -z "${GO2_IMAGE:-}" ] && docker image inspect khang/go2:dev >/dev/null 2>&1; then
        IMAGE=khang/go2:dev
        echo "Using existing image $IMAGE. A fresh machine builds go2:dev with ./controller/docker/build.sh"
    else
        echo "No Docker image named $IMAGE."
        echo "Build it once from this folder (needs a network connection, about 15-30 minutes):"
        echo "  ./controller/docker/build.sh"
        exit 1
    fi
fi

driver_ok() {
    docker exec $C bash -c "$ROS && timeout 4 ros2 topic hz /imu 2>/dev/null | grep -q average"
}

wait_for() {
    local secs=$1 what=$2; shift 2
    printf '%s' "$what"
    for _ in $(seq "$secs"); do
        if "$@" >/dev/null 2>&1; then echo " ok"; return 0; fi
        printf '.'; sleep 1
    done
    echo " timed out"; return 1
}

echo "This computer must be on the lab Wi-Fi (172.17.x). The dog joins that network on its own."
echo "Checking $DOG. Close the Unitree phone app first: the dog allows one connection."
wait_for 90 "Waiting for the dog on the network ($DOG)" ping -c1 -W1 $DOG \
    || { echo "Dog not reachable. Power it on, wait until it has joined the lab Wi-Fi, and confirm this computer is on that same Wi-Fi (not a phone hotspot)."; exit 1; }
wait_for 60 "Waiting for the dog's WebRTC port (9991)" nc -z -w1 $DOG 9991 \
    || { echo "The dog answers ping but port 9991 is closed, so the encrypted session cannot start. Wait a minute and try again."; exit 1; }

if docker inspect go2_test >/dev/null 2>&1 && [ "$(docker inspect -f '{{.State.Running}}' go2_test 2>/dev/null)" = "true" ]; then
    echo "The shared driver go2_test is still running (image rosrobot)."
    echo "Stop it before starting $C. The dog allows only one connection:"
    echo "  docker stop go2_test"
    exit 1
fi

if ! docker inspect $C >/dev/null 2>&1; then
    echo "Creating driver container from $IMAGE..."
    # --net=host: the container uses this computer's lab Wi-Fi address, so the
    # encrypted WebRTC session to the dog is a direct LAN connection.
    docker run -d --name $C --net=host --entrypoint bash \
        -v "$STACK/go2_recordings:/recordings" -v /etc/localtime:/etc/localtime:ro \
        -v "$STACK/go2_maps:/maps" -e MAP_NAME=lab \
        -e ROBOT_IP=$DOG -e CONN_TYPE=webrtc -e ROS_DOMAIN_ID=78 \
        -e GO2_CONTROL_NETS="$NETS" "$IMAGE" \
        -c "$ROS && ros2 launch go2_robot_sdk robot.launch.py rviz2:=false nav2:=false slam:=false foxglove:=false joystick:=false teleop:=false" >/dev/null
elif [ "$(docker inspect -f '{{.State.Running}}' $C 2>/dev/null)" != "true" ]; then
    echo "Starting driver..."; docker start $C >/dev/null
elif [ "${1:-}" = "--reset" ]; then
    echo "Restarting driver..."; docker restart $C >/dev/null
elif ! driver_ok; then
    echo "Driver not receiving data; restarting it..."; docker restart $C >/dev/null
else
    echo "Driver already connected."
fi

wait_for 45 "Waiting for sensor data" driver_ok || {
    echo "Driver still has no data. Close the Unitree app on all phones (only one connection"
    echo "is allowed), then run this again with --reset."; exit 1; }

for f in go2_keys.py go2_dashboard.py go2_mapper.py go2_trail.py; do
    docker cp -q "$(readlink -f "$STACK/$f")" $C:/root/$f
done
if ! docker exec $C pgrep -f go2_dashboard.py >/dev/null; then
    docker exec -d $C bash -c "$ROS && exec python3 /root/go2_dashboard.py 8080 > /tmp/dash.log 2>&1"
fi
URL=http://localhost:8080
wait_for 15 "Starting dashboard" curl -sf -o /dev/null $URL
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v xdg-open >/dev/null; then
    xdg-open $URL >/dev/null 2>&1 &
    echo "Opened the dashboard in your browser: $URL (tick Enable driving there to drive)"
else
    echo "Dashboard: $URL (no desktop session here, so open it yourself)"
fi
echo "Drive from the page with the keyboard (click the page first, tick Enable driving):"
echo "  hold W/S forward/back   A/D strafe   Q/E turn   space stop"
echo "Terminal keys, if you want them instead: docker exec -it $C bash -c \"$ROS && python3 /root/go2_keys.py\""
